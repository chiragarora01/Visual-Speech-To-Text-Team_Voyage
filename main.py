import argparse
import json
from collections import deque
from contextlib import contextmanager
from pathlib import Path
import time
import cv2
import face_alignment
import numpy as np
import torch
from torchvision.transforms.functional import to_tensor

from lipreading.model import Lipreading
from preprocessing.transform import warp_img, cut_patch

STD_SIZE = (256, 256)
STABLE_PNTS_IDS = [33, 36, 39, 42, 45]
START_IDX = 48
STOP_IDX = 68
CROP_WIDTH = CROP_HEIGHT = 96
s = ''
confidence = ''


@contextmanager
def VideoCapture(*args, **kwargs):
    cap = cv2.VideoCapture(*args, **kwargs)
    try:
        yield cap
    finally:
        cap.release()


def load_model(config_path: Path):
    with config_path.open() as fp:
        config = json.load(fp)
    tcn_options = {
        'num_layers': config['tcn_num_layers'],
        'kernel_size': config['tcn_kernel_size'],
        'dropout': config['tcn_dropout'],
        'dwpw': config['tcn_dwpw'],
        'width_mult': config['tcn_width_mult'],
    }
    return Lipreading(
        num_classes=500,
        tcn_options=tcn_options,
        backbone_type=config['backbone_type'],
        relu_type=config['relu_type'],
        width_mult=config['width_mult'],
        extract_feats=False,
    )


# s = ''.strip('\n')
# confidence = ''.strip('\n')
def visualize_probs(vocab, probs, col_width=4, col_height=300):
    num_classes = len(probs)
    out = np.zeros((col_height, num_classes * col_width + (num_classes - 1), 3), dtype=np.uint8)
    for i, p in enumerate(probs):
        x = (col_width + 1) * i
        cv2.rectangle(out, (x, 0), (x + col_width - 1, round(p * col_height)), (255, 255, 255), 1)
    top = np.argmax(probs)
    # s = ' '
    # confidence = ' '
    # print(probs)
    # print("==1==")
    # print(vocab[top])
    # print("==2==")
    # print(top)
    # print("==3==")
    # print(vocab[top])
    global s
    global confidence

    s = s + str(vocab[top]) + ' '
    t = str.join(" ", s.splitlines())
    confidence = confidence + str(probs[top]) + ' '
    # s=" ".join([s,str(vocab[top])])
    import os
    os.system('cls')
    print(t)
    print(confidence)
    # print( f'Prediction: {vocab[top]}\n')
    # print( f'Confidence: {probs[top]:.3f}')
    # global confidence
    # confidence = confidence + str(probs[top])
    # print(confidence,end = ' ')
    return t,confidence


def main(x):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path', type=Path, default=Path('training\lrw_resnet18_mstcn.json'))
    parser.add_argument('--model-path', type=Path, default=Path('training\lrw_resnet18_mstcn_adamw_s3.pth.tar'))
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--queue-length', type=int, default=29)
    args = parser.parse_args()

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device=args.device)
    model = load_model(args.config_path)
    model.load_state_dict(torch.load(Path(args.model_path), map_location=args.device)['model_state_dict'])
    model = model.to(args.device)

    mean_face_landmarks = np.load(Path('preprocessing/20words_mean_face.npy'))

    with Path('labels/500WordsSortedList.txt').open() as fp:
        vocab = fp.readlines()
    assert len(vocab) == 500

    queue = deque(maxlen=args.queue_length)
    prev_frame_time = 0

    # used to record the time at which we processed current frame
    new_frame_time = 0

    # -----1----------------------------------question brought by conservatives economic referendum-----------#
    # -----2----------------------------------children affected by emergency situations-----------#
    # -----3----------------------------------america announced public companion-----------#
    # -----4----------------------------------accused situation claims spending-----------#
    # print("enter 1 for camera\n enter 2 for video statement")
    # userchoice = int(input("Enter userchoice:"))
    with VideoCapture(x) as cap:
        while True:
            ret, image_np = cap.read()
            if not ret:
                break
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

            all_landmarks = fa.get_landmarks(image_np)
            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time
            fps = int(fps)
            fps = str(fps)
            # print("fps="+fps)
            if all_landmarks:
                landmarks = all_landmarks[0]

                # BEGIN PROCESSING

                trans_frame, trans = warp_img(
                    landmarks[STABLE_PNTS_IDS, :], mean_face_landmarks[STABLE_PNTS_IDS, :], image_np, STD_SIZE)
                trans_landmarks = trans(landmarks)
                patch = cut_patch(
                    trans_frame, trans_landmarks[START_IDX:STOP_IDX], CROP_HEIGHT // 2, CROP_WIDTH // 2)

                cv2.imshow('patch', cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))

                patch_torch = to_tensor(cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)).to(args.device)
                queue.append(patch_torch)
                print(args.queue_length)
                print(len(queue))
                if len(queue) >= args.queue_length:
                    with torch.no_grad():
                        model_input = torch.stack(list(queue), dim=1).unsqueeze(0)
                        logits = model(model_input, lengths=[args.queue_length])
                        probs = torch.nn.functional.softmax(logits, dim=-1)
                        probs = probs[0].detach().cpu().numpy()
                        queue.clear()
                    vis,conf = visualize_probs(vocab, probs)
                    # cv2.imshow('probs', vis)
                    # print("vis",vis)
                    # print("conf",conf)
                # END PROCESSING

                for x, y in landmarks:
                    cv2.circle(image_np, (int(x), int(y)), 2, (0, 0, 255))

            cv2.imshow('camera', cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

            key = cv2.waitKey(1)
            if key in {27, ord('q')}:  # 27 is Esc
                break
            elif key == ord(' '):
                cv2.waitKey(0)
        cv2.destroyAllWindows()
    return vis,conf


# if name == 'main':
#     main()

############################################################################
from tkinter import *
from tkinter.filedialog import askopenfile
from tkinter import font as tkFont

w = Tk()
w.geometry('1000x600')
w.configure(bg='#141414')
w.title("Visual Speech recognition (Lip-Reading)")

helv = tkFont.Font(family='colortube', size=16, weight='bold')


def cmd():
    main(0)
    print('You Clicked Camera')


# def vid():
#     print('You Clicked Video')


def open_file():
    file_path = askopenfile(mode='r', filetypes=[('Image Files', '*mp4')])
    print(file_path.name)
    output_text,output_confidence=main(file_path.name)
    print("output_text",output_text.strip('\n'))
    print("output_confidence", output_confidence)
    print(file_path)
    
    l2=Label(w, text=output_confidence)
    l = Label(w, text=output_text)
    # l.destroy()
    # l2.destroy()
    l.config(font=("Courier", 14), bg="#141414", fg="#ffcc66")
    l2.config(font=("Courier", 14), bg="#141414", fg="#ffcc66")
    l.place(relx=0.32, rely=0.8, anchor="sw")
    l2.place(relx=0.32, rely=0.9, anchor="sw")
    # import time
    # time.sleep(1000)
    # l.destroy()
    # l2.destroy()
    global s
    global confidence
    s=''
    confidence=''
    if file_path is not None:
        pass


def bttn(x, y, text, bcolor, fcolor, cmd, font):
    def on_enter(e):
        mybutton['background'] = bcolor
        mybutton['foreground'] = fcolor

    def on_leave(e):
        mybutton['background'] = fcolor
        mybutton['foreground'] = bcolor

    mybutton = Button(w, width=22, height=11, text=text, fg=bcolor, bg=fcolor, border=1, activebackground=fcolor,
                      activeforeground=bcolor, command=cmd, font=font)

    mybutton.bind("<Enter>", on_enter)
    mybutton.bind("<Leave>", on_leave)
    mybutton.place(x=x, y=y)



bttn(200, 137, "Video", "#ffcc66", "#141414", open_file, helv)
bttn(500, 137, "Camera", "#25dae9", "#141414", cmd, helv)
w.mainloop()