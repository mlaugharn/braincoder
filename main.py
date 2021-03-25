from preparation import *

gen_npz()
handler, Y, Y_test, Y_test_avg, labels_train, labels, NUM_VOXELS, images, X, X_test, train_dl, test_dl = prepare_setup()
xyz, dim = handler.get_voxel_loc()

# import IPython
# IPython.embed()

from encoder import FineTuneModel
import torchvision

original_model = torchvision.models.resnet34(pretrained=True)
arch = 'resnet34'
encoder = FineTuneModel(original_model, arch, NUM_VOXELS=NUM_VOXELS, xyz=xyz).cuda()

import torch.optim as optim
import torch.nn as nn
import torch
import einops
from tqdm.autonotebook import tqdm



def train_encoder_model(encoder, train_dl, test_dl, Y, Y_test, epochs=config.num_enc_epochs):
    epochs_losses = []
    criterion = nn.MSELoss()
    optimizer = optim.Adam(lr=1e-3, params=encoder.parameters())
    
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    Y = torch.from_numpy(Y).float().cuda()
    Y_test = torch.from_numpy(Y_test).float().cuda()
    epochs_losses_test = []
    for epoch in tqdm(range(epochs)):
        losses = []
        test_losses = []
        encoder.train()
        
        for i, batch in enumerate(train_dl, 0):
            inputs, mris, idxs = batch
            inputs = einops.rearrange(inputs, 'b h w c -> b c h w')
#             inputs.cuda()
            Y_batch = mris.float().cuda()
            with torch.cuda.amp.autocast():
                outputs = encoder(inputs.float().cuda()).float().cuda()
#             print(len(inputs), len(idxs))
#             print(outputs.size())
#             print(outputs)
#             print(Y_batch)
                loss = criterion(outputs, Y_batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # scheduler.step()

            optimizer.zero_grad()
            losses.append(loss.item())
        encoder.eval()
        with torch.no_grad():
            for i, batch in enumerate(test_dl):
                inputs, mris, idxs = batch
                inputs = einops.rearrange(inputs, 'b h w c -> b c h w')
                Y_test_batch = mris.float().cuda()
                with torch.cuda.amp.autocast():
                    outputs = encoder(inputs.float().cuda()).float().cuda()
                    loss = criterion(outputs, Y_test_batch)

                test_losses.append(loss.item())
        print(f'[{epoch+1}, loss: {np.mean(losses):.4f} test loss: {np.mean(test_losses):.4f}')
        epochs_losses.append(np.mean(losses))
        epochs_losses_test.append(np.mean(test_losses))
        losses = []
        test_losses =[]
    # import matplotlib.pyplot as plt
    # plt.plot(range(epochs), epochs_losses, label='train')
    # plt.plot(range(epochs), epochs_losses_test, label='test')
    # plt.legend()
    # plt.show()
    return encoder

encoder = train_encoder_model(encoder, train_dl, test_dl, Y, Y_test_avg)

import decoder
import pytorch_ssim

# decoder = decoder.make_decoder(NUM_VOXELS).cuda()
# decoder = decoder.make_original_decoder(NUM_VOXELS).cuda()
decoder = decoder.new_decoder(NUM_VOXELS).cuda()

import random
import lpips
import piq
import piqa
skip = False
def train_simultaneous_decoder_objectives(encoder, decoder, train_dl, test_dl, Y, Y_test, Y_test_avg, epochs=config.num_multi_epochs):
    global NUM_VOXELS
    # encoder.eval()
    encoder.eval()
    encoder.trainable = False
    decoder.train()
    decoder.trainable = True
    print(decoder)
    Y = torch.from_numpy(Y).float()#.cuda()
    # Y = Y.reshape(-1, NUM_VOXELS, 1, 1) # turn fmri into 1 x NUMVOXELS grayscale image
    Y_test = torch.from_numpy(Y_test).float()
    Y_test_avg = torch.from_numpy(Y_test_avg).float()#.cuda()
    test_fmri_dl = make_test_fmri_dl(Y_test_avg)

    msecriterion = nn.MSELoss()
    # maecriterion = nn.L1Loss()
    # ssimcriterion = piq.SSIMLoss(data_range=1.)#
    # ssimcriterion = pytorch_ssim.SSIM()
    # perceptualcriterion = lpips.LPIPS(net='alex').cuda()
    # mdsicriterion = piqa.MDSI().cuda()
    coscriterion = nn.CosineSimilarity()

    # enc_optim = optim.AdamW(lr=0, params=encoder.parameters())
    optimizer = optim.Adam(lr=1e-3, params=list(decoder.parameters()) # + list(encoder.parameters())
        , weight_decay=1e-3)
    
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)

    epoch_dec_losses = []
    epoch_decenc_losses = []
    epoch_encdec_losses = []

    imagenet = imagenet_dl()
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    objectives = ["d"] * 80 + ["ed"] * 20 # ["d"] * 60 + ["de"] * 10 + ["ed"] * 30 + ["gan"] * 0
    for epoch in tqdm(range(epochs)):
        decoder.trainable = True
        decoder.train()
        dec_losses = []
        decenc_losses = []
        encdec_losses = []

        for i, batch in enumerate(train_dl):
            # TODO:
            # - use test set of MRIs in decenc
            # - transformer decoder?
            # - imagenet val set in encdec

            inputs, mris, idxs = batch
            batch_size = len(inputs)
            inputs=inputs.permute(0,3,1,2).float().cuda()
            # Y_batch = Y[idxs].cuda() # if we go to next batch in train_dl, but then pick random objective of training decenc on testset statistics fmri, we're going to be biased against training on the trainset mri->mri mapping..
            Y_batch = mris.float().cuda()

            # not sure why it is so memory intensive to do all 3.. doing a random choice of any
            objective = random.choice(objectives) if epoch > 0 else "d"


            # enc_optim.zero_grad()            

                # dec:
                # D: fMRI -> image

            if objective == "d":
                with torch.cuda.amp.autocast():
                    dec_outputs = decoder(Y_batch).float().cuda() # [b, c, h, w]
                    # print(dec_outputs.shape, inputs.shape)
                    dec_loss = msecriterion(dec_outputs, inputs) 
                    # dec_loss = mdsicriterion(dec_outputs, inputs.permute(0, 3, 1, 2))
                    # dec_loss += maecriterion(dec_outputs, inputs)# + ssimcriterion(dec_outputs, inputs.permute(0, 3, 1, 2))
                    # dec_loss -= ssimcriterion(dec_outputs, inputs)
                    # dec_loss += perceptualcriterion(dec_outputs, inputs)
                    # perceptualloss = perceptualcriterion.forward(dec_outputs, inputs, normalize=True).cuda()
                    # dec_loss += 0.01 * torch.sum(perceptualloss)
                    # msecriterion(dec_outputs.permute(0, 2, 3, 1), inputs) + -ssimcriterion(dec_outputs.permute(0, 2, 3, 1), inputs) \
                        # 
                        
                    loss = dec_loss
                    dec_losses.append(dec_loss.item())
            # print("d", dec_outputs.permute(0, 2, 3, 1).shape, inputs.shape)
                
                # decenc: 
                # E . D: mri -> mri
            elif objective == "de":
                fmri_set = random.choice(["trainset", "testset"])
                if fmri_set == "testset":
                    print(">testset fmri")
                    del Y_batch
                    Y_batch = next(iter(test_fmri_dl)).float().cuda()
                with torch.cuda.amp.autocast():
                    dec_outputs = decoder(Y_batch).float().cuda() # [b, c, h, w]
                    decenc_outputs = encoder(dec_outputs)#.reshape(batch_size, NUM_VOXELS, 1, 1)
                    decenc_loss = msecriterion(decenc_outputs, Y_batch) 
                    decenc_loss += (1-torch.mean(coscriterion(decenc_outputs, Y_batch)))
                    loss = decenc_loss
                    decenc_losses.append(decenc_loss.item())
            # print("de", decenc_outputs.shape, Y_batch.shape)

                # encdec:
                # D. E: img -> img
            
            elif objective == "ed":
            # enc: b h w c -> b c h w ->
            # dec then b c h w -> b h w c
                img_src = random.choice(["trainset", "trainset", "imagenet"])
                if img_src == "imagenet":
                    print(">imagenet batch")
                    del inputs
                    inputs = next(iter(imagenet)).float().cuda()
                
                with torch.cuda.amp.autocast():
                    encdec_outputs = decoder(
                        encoder(inputs)#.reshape(batch_size, NUM_VOXELS, 1, 1)
                        )
                    encdec_loss = msecriterion(encdec_outputs, inputs)
                    # encdec_loss += perceptualcriterion(dec_outputs, inputs)
                    # encdec_loss = mdsicriterion(encdec_outputs, inputs.permute(0, 3, 1, 2))
                    # encdec_loss += maecriterion(encdec_outputs, inputs)
                    # encdec_loss -= ssimcriterion(encdec_outputs, inputs)
                    # encdec_loss = contentloss(encdec_outputs, inputs.permute(0, 3, 1, 2))
                        # msecriterion(encdec_outputs, inputs) -ssimcriterion(encdec_outputs, inputs) 
                        
                        
                    loss = encdec_loss
                    encdec_losses.append(encdec_loss.item())
            # print("ed", encdec_outputs.shape, inputs.shape)

            elif objective == "gan":
                pass

            # loss = torch.sum(dec_loss) + torch.sum(decenc_loss) + torch.sum(encdec_loss)
            # scaled_grad_params = torch.autograd.grad(outputs=scaler.scale(loss),
            #                                         inputs=decoder.parameters(),
            #                                         create_graph=True
            # )

            # inv_scale = 1./scaler.get_scale()
            # grad_params = [p * inv_scale for p in scaled_grad_params]

            # with torch.cuda.amp.autocast():
            #     grad_norm = 0
            #     for grad in grad_params:
            #         grad_norm += grad.pow(2).sum()
            #     grad_norm = grad_norm.sqrt()
            #     loss = loss + grad_norm

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # scheduler.step()

            optimizer.zero_grad()
            # enc_optim.step()

            
        print(f"epoch {epoch} mri->img: {np.mean(dec_losses)}  mri->mri: {np.mean(decenc_losses)}   img->img: {np.mean(encdec_losses)}")
        epoch_dec_losses.append(np.mean(dec_losses))
        epoch_decenc_losses.append(np.mean(decenc_losses))
        epoch_encdec_losses.append(np.mean(encdec_losses))

        # if epoch % 5 == 0:
        #     with torch.no_grad():
        #         eval_decoder(decoder, test_dl, X_test, Y_test_avg, avg=True)
        
        if epoch % 20 == 0:
            print("running through whole un-averaged testset")
            with torch.no_grad():
                # eval_decoder(decoder, test_dl, X_test, Y_test_avg, Y_test=Y, avg=False)
                decode_test_set(test_dl, Y_test, X, decoder)
        
        if epoch % 20 == 0:
            print("dumping trainset results")
            with torch.no_grad():
                decode_training_set(train_dl, Y, X, decoder)

    import matplotlib.pyplot as plt
    # plt.plot(range(len(epoch_dec_losses)), epoch_dec_losses, label='dec')
    # plt.plot(range(len(epoch_decenc_losses)), epoch_decenc_losses, label='decenc')
    # plt.plot(range(len(epoch_encdec_losses)), epoch_encdec_losses, label='encdec')
    # plt.legend()
    # plt.show()
    return decoder

import torch.autograd.profiler as profiler

from PIL import Image

def eval_decoder(decoder, test_dl, X_test, Y_test_avg, Y_test = None, avg=False):
    decoder.eval()
    Y_test_avg = Y_test_avg.float().cuda()

    inv_normalize = torchvision.transforms.Compose([T.ToTensor(), 
        torchvision.transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.255]
    )]) if config.normalize else torchvision.transforms.Compose([T.ToTensor()])

    if avg:
        quick_lookup_hashmap = {} # i (1..50) -> X_i 
        for i in range(50):
            for x_idx, x in enumerate(X_test):
                if labels[x_idx] == i:
                    quick_lookup_hashmap[i] = x
                    break
        batch_input = Y_test_avg
        stacked_imgs = np.array([quick_lookup_hashmap[i] for i in range(50)])
        stacked_imgs = stacked_imgs
        print('img stack shape', stacked_imgs.shape)
        gt_batch = np.array([inv_normalize(stacked_img).permute(1, 2, 0).numpy() for stacked_img in stacked_imgs])
        with torch.cuda.amp.autocast():
            preds = decoder(batch_input).detach().cpu().permute(0, 2, 3, 1).numpy()

        print("shapes: ", gt_batch.shape, preds.shape)

        for i in range(50):
            # mri = Y_test_avg[i]
            # gt_img = quick_lookup_hashmap[i]#test_dl.dataset[i][0]
            # batch = mri.reshape(1, NUM_VOXELS).float().cuda()
            
            # pred_img = inv_normalize(decoder(batch))[0].permute(1, 2, 0).detach().cpu().numpy()
            gt_img = gt_batch[i]
            pred_img = preds[i]
            
            combo = np.clip(np.hstack((gt_img, pred_img)) * 255, 0, 255)
            combo = combo.astype(np.uint8)

            combo_im = Image.fromarray(combo)
            combo_im.save(f"combo{i}.png")
    else:
        all_idxs = list(range(len(Y_test)))
        for batch_i in range(0, len(Y_test), config.batch_size):
            idxs = all_idxs[batch_i: batch_i + config.batch_size]
            imgs = torch.from_numpy(X_test[idxs])
            imgs = imgs.permute(0,3,1,2).float()
            print(imgs.shape, " imgs")
            imgs = inv_normalize(imgs).permute(0, 2, 3, 1).numpy()
            Y_batch = torch.from_numpy(Y[idxs]).cuda().float().cuda()
            with torch.cuda.amp.autocast():
                preds = decoder(Y_batch).permute(0, 2, 3, 1)
            print(preds.shape, " preds")
            preds = preds.detach().cpu().numpy()
            combos = np.hstack((imgs, preds))
            for combo, idx in zip(combos, idxs):
                combo = np.clip(combo * 255, 0, 255).astype(np.uint8)
                combo_im = Image.fromarray(combo)
                combo_im.save(f"y_not_avg/combo{idx}.png")

def decode_test_set(test_dl, Y_test, X, decoder):
    inv_normalize = T.Compose([torchvision.transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.255])]) if config.normalize else torchvision.transforms.Compose([])
    for i, batch in enumerate(test_dl):
        imgs, mris, idxs = batch
        imgs = imgs#.permute(0,3,1,2)
        # mris = Y_test[idxs]
        mris = mris.float().cuda()
        preds = decoder(mris).permute(0,2,3,1).detach().cpu().numpy()
        for img, pred, idx in zip(imgs, preds, idxs):
            combo = torch.from_numpy(np.clip(np.hstack((img, pred)) * 255, 0, 255))
            combo = inv_normalize(combo.permute(2,0,1)).permute(1,2,0).numpy()
            # combo = combo.numpy() if isinstance(combo, torch.Tensor) else combo
            combo = combo.astype(np.uint8)
            combo_im = Image.fromarray(combo)
            combo_im.save(f"testsetresults/combo{idx}.png")


def decode_training_set(train_dl, Y, X, decoder):
    inv_normalize = T.Compose([torchvision.transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.255])]) if config.normalize else torchvision.transforms.Compose([])
    for i, batch in enumerate(train_dl):
        imgs, mris, idxs = batch
        imgs = imgs#.permute(0,3,1,2)
        # mris = Y[idxs]
        mris = mris.float().cuda()
        preds = decoder(mris).permute(0,2,3,1).detach().cpu().numpy()
        for img, pred, idx in zip(imgs, preds, idxs):
            combo = torch.from_numpy(np.clip(np.hstack((img, pred)) * 255, 0, 255))
            combo = inv_normalize(combo.permute(2,0,1)).permute(1,2,0).numpy()
            # combo = combo.numpy() if isinstance(combo, torch.Tensor) else combo
            combo = combo.astype(np.uint8)
            combo_im = Image.fromarray(combo)
            combo_im.save(f"trainsetresults/combo{idx}.png")


profile = False
if profile:
    with profiler.profile(record_shapes=True, profile_memory=True, use_cuda=True) as prof:
        with profiler.record_function("model_inference"):
            decoder = train_simultaneous_decoder_objectives(encoder, decoder, train_dl, test_dl, Y, Y_test)
else:
    decoder = train_simultaneous_decoder_objectives(encoder, decoder, train_dl, test_dl, Y, Y_test, Y_test_avg)

import IPython
IPython.embed()


try:
    eval_decoder(decoder, test_dl, X_test, torch.from_numpy(Y_test_avg))
except:
    IPython.embed()


# def decode_training_set(train_dl, Y, X, decoder):
#     for i, batch in enumerate(train_dl):
#         imgs, idxs = batch
#         mris = Y[idxs]
#         preds = decoder(mris)
#         for img, pred in zip(imgs, preds):
#             combo = ..
#             savecombo()


import IPython
IPython.embed()