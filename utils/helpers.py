from torch.utils.tensorboard import SummaryWriter


def write_logs_tb(experiment, img_fake, img_real, loss_D, loss_G, step, epoch, hyperparams, with_print_logs=True):

    tb_writer_fake = SummaryWriter(f"logs/{experiment}_GAN/fake_{experiment}")
    tb_writer_real = SummaryWriter(f"logs/{experiment}_GAN/real_{experiment}")
    tb_writer_loss = SummaryWriter(f"logs/{experiment}_GAN/loss_train_{experiment}")

    # Adding loss values to tb
    tb_writer_loss.add_scalar(
        "loss_D", loss_D, global_step=step
    )

    tb_writer_loss.add_scalar(
        "loss_G", loss_G, global_step=step
    )
        
    # Adding generated images to tb
    tb_writer_fake.add_image(
        "Fake images", img_fake, global_step=step
    )

    tb_writer_real.add_image(
        "Real images", img_real, global_step=step
    )

    if with_print_logs :
        print(
            f"Epoch [{epoch}/{hyperparams.n_epochs}] \ "
            f"Loss_D : [{loss_D:.4f}] \ "
            f"loss_G : [{loss_G:.4f}] \ "
        )