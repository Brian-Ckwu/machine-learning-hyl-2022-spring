import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

def plot_learning_curve(loss_record, title="Learning Curve", xlabel="Training steps", ylabel="MSE loss"):
    # set x-axis
    total_steps = len(loss_record["train"])
    x_train = range(total_steps)
    val_step_width = total_steps // len(loss_record["val"])
    x_val = x_train[::val_step_width]
    
    # set figure
    figure(figsize=(12, 8))
    plt.plot(x_train, loss_record["train"], c="tab:red", label="train")
    plt.plot(x_val, loss_record["val"], c="tab:cyan", label="val")
    plt.ylim(0.0, 5.0)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    # plot figure
    plt.legend()
    plt.show()
