import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection, LineCollection
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


def image_pt_to_np(image):
    """
    Convert a PyTorch image to a NumPy image (in the OpenCV format).
    """
    image = image.cpu().clone()
    image = image.permute(1, 2, 0)
    image_mean = torch.tensor([0.485, 0.456, 0.406])
    image_sd = torch.tensor([0.229, 0.224, 0.225])
    image = image * image_sd + image_mean
    image = image.clip(0, 1)
    return image


def show_warps(warps, nrow=8, fname=None, show=False):
    """
    Plot a tensor of image patches / warps.
    """
    image_grid = torchvision.utils.make_grid(warps, nrow=nrow)
    fig, ax = plt.subplots(figsize=[8, 8])
    ax.imshow(image_pt_to_np(image_grid))
    ax.axis('off')
    save_fig(fig, fname, show)
    

def occupancy_colors(scores):
    """
    Set the coloring scheme for occupancy plots.
    """
    colors = np.zeros([len(scores), 3])
    colors[:, 0] = scores     # red
    colors[:, 1] = 1 - scores # green
    colors[:, 2] = 0          # blue
    return colors


root = tk.Tk()

root.geometry("1200x800")

# Create a canvas to display the Matplotlib figure
canvas = None

def save_fig(fig, fname, Occupied_count, UnOccupied_count, show=False):
    """
    A helper function to show or save a figure.
    """
    global canvas, occupiedCount_label, unoccupiedCount_label
    
    # Create a Tkinter window
    if canvas is None:
        root.geometry('1200x800')
        root.title("Occupancy Visualization")
        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas.get_tk_widget().configure(width=1200, height=800)
        canvas.draw()
        canvas.get_tk_widget().pack(expand=True, fill=tk.BOTH)
        
        # Create a Label widget to display the count
        occupiedCount_label = tk.Label(root, text="Occupied count: {}".format(Occupied_count), font=("Roboto Bold", 20))
        occupiedCount_label.place(x=10, y=10)
        unoccupiedCount_label = tk.Label(root, text="Unoccupied count: {}".format(UnOccupied_count), font=("Roboto Bold", 20))
        unoccupiedCount_label.place(x=10, y=60)
        
        # Create a button to close the Tkinter window
        close_button = tk.Button(root, text="Close", command=root.destroy)
        
    else:
        # Update the existing canvas with the new figure
        canvas.figure = fig
        canvas.draw_idle()
        
        # Update the count label
        occupiedCount_label.configure(text="Occupied count: {}".format(Occupied_count))
        unoccupiedCount_label.configure(text="Unoccupied count: {}".format(UnOccupied_count))

        #Close on click of closing button


    # Update the Tkinter event loop to display the updated figure and count
    root.update() 


root.resizable(True, True)

# Bind the <KeyPress> event to the root window
def handle_keypress(event):
    if event.char == 'q':
        root.destroy()

root.bind("<KeyPress>", handle_keypress)


def plot_ds_image(image, rois, occupancy, true_occupancy=None, fname=None, show=False):
    """
    Plot an annotated dataset image with occupancy equal to `occupancy`.
    If `true_occupancy` is specified, `occupancy` is assumed to represent the
    predicted occupancy.
    """

    # plot image
    fig, ax = plt.subplots(figsize=[12, 8])
    ax.imshow(image_pt_to_np(image))
    ax.axis('off')
    
    # convert rois
    C, H, W = image.shape
    rois = rois.cpu().clone()
    rois[..., 0] *= (W - 1)
    rois[..., 1] *= (H - 1)
    rois = rois.numpy()
    # add root array index as text label
    ax.text(0, 0, "Root Array Index: {}".format(str(0)), ha='left', va='bottom', color='white', fontsize=12)
    
    # plot annotations
    polygons = []
    colors = occupancy_colors(occupancy.detach().cpu().numpy())
    
    for i, (roi, color) in enumerate(zip(rois, colors)):
        polygon = Polygon(roi, fc=color, alpha=0.3)
        polygons.append(polygon)
        
        # add ROI index as text label
        x = roi[:, 0].mean()
        y = roi[:, 1].mean()
        ax.text(x, y, str(i), ha='center', va='center', color='white')
    p = PatchCollection(polygons, match_original=True)
    ax.add_collection(p)

    #Count the number of occupied ROIs
    occupied_count = np.count_nonzero(occupancy.detach().cpu().numpy() >= 0.5)
    unOccupied_count = np.count_nonzero(occupancy.detach().cpu().numpy() < 0.5)
    
    # plot prediction
    if true_occupancy is not None:
        # only show those crosses where the predictions are incorrect
        pred_inc = occupancy.round() != true_occupancy
        rois_subset = rois[pred_inc]
        ann_subset = true_occupancy[pred_inc]
        
        # create an array of crosses for each parking space
        lines = np.array(rois_subset)[:, [0, 2, 1, 3], :].reshape(len(rois_subset)*2, 2, 2)
        colors = occupancy_colors(ann_subset.cpu().numpy())
        
        # add the crosses to the plot
        colors = np.repeat(colors, 2, axis=0)
        lc = LineCollection(lines, colors=colors, lw=1)
        ax.add_collection(lc)
    
    # save figure
    # save_fig(fig, fname, occupied_count, unOccupied_count, show)
    root.after(33, save_fig(fig, fname, occupied_count, unOccupied_count, show))
