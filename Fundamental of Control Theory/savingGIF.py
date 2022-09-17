# -*- coding: utf-8 -*-
"""
Created on Sun Jul 17 08:02:37 2022

@author: 86130
"""
from matplotlib import animation
import matplotlib.pyplot as plt

def save_frames_as_gif(frames, index,path='./'):
    filename = 'gym_animation' + str(index) + '.gif'
    path = path + filename
    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path, writer='pillow', fps=60)