import numpy as np
import matplotlib.pyplot as plt

def plot_deformation_grid(pos_field, out_path, step=16):
        pos_field = pos_field[0]  # first batch
        pos_field = pos_field.detach().cpu().numpy()  # [2, H, W]
        H, W = pos_field.shape[1:]

        grid_y, grid_x = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        
        new_y = grid_y + pos_field[0]
        new_x = grid_x + pos_field[1]
        
        plt.figure(figsize=(8, 8))
        for i in range(0, H, step):
            plt.plot(new_x[i, ::step], new_y[i, ::step], 'b-')
        for j in range(0, W, step):
            plt.plot(new_x[::step, j], new_y[::step, j], 'b-')
        plt.gca().invert_yaxis()
        plt.title("Deformation Field Grid")
        plt.axis('off')
        plt.savefig(out_path, bbox_inches='tight')
        plt.close()


def plot_deformation_heatmap(pos_field, out_path):
    pos_field = pos_field[0].detach().cpu().numpy()  # [2, H, W]
    magnitude = np.sqrt(np.sum(pos_field**2, axis=0))  # [H, W]
    plt.figure(figsize=(8, 8))
    plt.imshow(magnitude, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Deformation Magnitude')
    plt.title('Deformation Field Magnitude Heatmap')
    plt.axis('off')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()