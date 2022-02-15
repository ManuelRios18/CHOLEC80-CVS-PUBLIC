from colenet.trainer import ColenetTrainer

root_dir = "/media/manuel/DATA/datasets/COLELAPS FRAMES"
backbone = "vgg"
log_name = "vgg"

epochs = 10
batch_size = 32
learning_rate = 1e-5

trainer = ColenetTrainer(root_dir, backbone, log_name, "mean_f1")
trainer.train_colenet(epochs, batch_size, learning_rate)
