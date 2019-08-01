import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as ag
import argparse
from Datasets import *
import matplotlib.pyplot as plt
import os
import itertools
from torch.optim import lr_scheduler


class Generator(nn.Module):
    def __init__(self, nhidden, nlayers):
        super(Generator, self).__init__()
        self.net = nn.Sequential()
        self.net.add_module('input', nn.Linear(2, nhidden))
        self.net.add_module('act0', nn.ReLU(True))
        for i in range(nlayers):
            self.net.add_module('hidden_%d' % (i + 1), nn.Linear(nhidden, nhidden))

            self.net.add_module('act_%d' % (i + 1), nn.ReLU(True))
        self.net.add_module('output', nn.Linear(nhidden, 2))

    def forward(self, z):
        return self.net(z)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Critic(nn.Module):
    def __init__(self, nhidden, nlayers):
        super(Critic, self).__init__()
        # self.net = nn.Sequential(
        #     nn.Linear(2, nhidden),
        #     nn.ReLU(True),
        #     nn.Linear(nhidden, nhidden),
        #     nn.ReLU(True),
        #     nn.Linear(nhidden, nhidden),
        #     nn.ReLU(True),
        #     nn.Linear(nhidden, 1),
        # )
        self.net = nn.Sequential()
        self.net.add_module('input', nn.Linear(2, nhidden))
        self.net.add_module('act0', nn.ReLU(True))
        for i in range(nlayers):
            self.net.add_module('hidden_%d' % (i + 1), nn.Linear(nhidden, nhidden))
            self.net.add_module('act_%d' % (i + 1), nn.ReLU(True))
        self.net.add_module('output', nn.Linear(nhidden, 1))

    def forward(self, x):
        return self.net(x).view(-1)



class Criticr_realtion2(nn.Module):
    def __init__(self,nhidden,nlayers):
        super(Criticr_realtion2, self).__init__()
        self.net = nn.Sequential()
        self.net.add_module('input', nn.Linear(2, nhidden))
        self.net.add_module('act0', nn.ReLU(True))
        for i in range(nlayers):
            self.net.add_module('hidden_%d' % (i + 1), nn.Linear(nhidden, nhidden))
            self.net.add_module('act_%d' % (i + 1), nn.ReLU(True))
        self.re = nn.Sequential()
        self.re.add_module('re',nn.Linear(2*nhidden,1))
    def forward(self,x1,x2):
        em1 = self.net(x1)
        em2 = self.net(x2)
        re = self.re(torch.cat([em1,em2],dim=1))
        return re



class Criticr_pac2(nn.Module):
    def __init__(self,nhidden,nlayers):
        super(Criticr_pac2, self).__init__()
        self.net = nn.Sequential()
        self.net.add_module('input', nn.Linear(2, nhidden))
        self.net.add_module('act0', nn.ReLU(True))

        self.re = nn.Sequential()
        self.re.add_module('re',nn.Linear(2*nhidden,nhidden))
        self.re.add_module('act', nn.ReLU(True))
        self.re.add_module('hidden_', nn.Linear(nhidden, 1))

    def forward(self,x1,x2):
        em1 = self.net(x1)
        em2 = self.net(x2)
        re = self.re(torch.cat([em1,em2],dim=1))

        return re

class Criticr_realtion3(nn.Module):
    def __init__(self,nhidden,nlayers):
        super(Criticr_realtion3, self).__init__()
        self.net = nn.Sequential()
        self.net.add_module('input', nn.Linear(2, nhidden))
        self.net.add_module('act0', nn.ReLU(True))
        for i in range(nlayers):
            self.net.add_module('hidden_%d' % (i + 1), nn.Linear(nhidden, nhidden))
            self.net.add_module('act_%d' % (i + 1), nn.ReLU(True))
        self.re = nn.Sequential()
        self.re.add_module('re',nn.Linear(2*nhidden,nhidden))
        self.linear=nn.Linear(3*nhidden,2)
    def forward(self,x1,x2,comp):
        em1 = self.net(x1)
        em2 = self.net(x2)
        comp3 = self.net(comp)
       # re1 = self.re(torch.cat([em1,comp3]))
       # re2 = self.re(torch.cat([em2,comp3]))


        score = self.linear(torch.cat([em1,em2,comp3],dim=1))

        return score

def GAN_GP(D, G, data, noise, niter=10000, batch_size=32, optimizer='Adam',
           lrg=1e-3, lrd=3e-3, center=0, LAMBDA=1, alpha=None, device='cuda', prefix='figs/', gp=False,args=None):
    D.to(device)
    G.to(device)

    if optimizer == 'SGD':
        optim_d = optim.SGD(D.parameters(), lr=lrd)
        optim_g = optim.SGD(G.parameters(), lr=lrg)
    elif optimizer == 'Adam':
        optim_d = optim.Adam(D.parameters(), lr=lrd, betas=(0.5, 0.9))
        optim_g = optim.Adam(G.parameters(), lr=lrg, betas=(0.5, 0.9))
    elif optimizer == 'RMS':
        optim_d = optim.RMSprop(itertools.chain(D.parameters()), lr=lrd, alpha=0.9)
        optim_g = optim.RMSprop(itertools.chain(G.parameters()), lr=lrg, alpha=0.9)
    criterion = nn.MSELoss()

    zeros = torch.zeros(batch_size, device=device)
    ones = torch.ones(batch_size, device=device)

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))

    scale = 1
    if args is not None:
        scale = args.scale
    scale *= data.range

    for iter in range(niter):
        if iter % 100 == 0:
            print(iter)

            noise_batch = noise.next_batch(512, device=device)
            fake_batch = G(noise_batch)
            fake_batch = fake_batch.data.cpu().numpy()

            real_batch = data.next_batch(512, device=device)

            plt.figure(fig.number)
            ax.clear()
            ax.scatter(real_batch[:, 0].cpu(), real_batch[:, 1].cpu(), s=2)
            ax.scatter(fake_batch[:, 0], fake_batch[:, 1], s=2, c='r', marker='+')
            ax.set_xlim((-scale, scale))
            ax.set_ylim((-scale, scale))

            coord, grad = visualize_grad(G, D, criterion, fig, ax, scale=scale, device=device)
            plt.draw()
            plt.savefig(prefix + 'fig_%05d.pdf' % iter, bbox_inches='tight')
            plt.pause(0.1)

        # train D
        optim_d.zero_grad()
        real_batch = data.next_batch(batch_size, device=device)
        predict_real = D(real_batch)
        loss_real = criterion.forward(predict_real, ones)

        noise_batch = noise.next_batch(batch_size, device=device)
        fake_batch = G(noise_batch)
        fake_batch = fake_batch.detach()
        predict_fake = D(fake_batch)
        loss_fake = criterion.forward(predict_fake, zeros)
        if gp:
            gradpen = cal_gradpen(D, real_batch.detach(), fake_batch.detach(),
                              center=center, LAMBDA=LAMBDA, alpha=alpha, device=device)
            loss_d = loss_real + loss_fake + gradpen
        else:
            loss_d = loss_real + loss_fake
        loss_d.backward()
        optim_d.step()

        # train G
        optim_g.zero_grad()
        noise_batch = noise.next_batch(batch_size)
        noise_batch = noise_batch.to(device)
        fake_batch = G(noise_batch)
        predict_fake = D(fake_batch)
        loss_g = criterion.forward(predict_fake, ones)
        loss_g.backward()
        optim_g.step()
        print('d',float(loss_d),'g:',float(loss_g))

    return D, G

def GAN_GP_RE2(D, G, data, noise, niter=10000, batch_size=32, optimizer='Adam',
           lrg=1e-3, lrd=3e-3, center=0, LAMBDA=1, alpha=None, device='cuda', prefix='figs/', args=None,gp=False):
    D.to(device)
    G.to(device)
    #D.apply(weights_init)
    #G.apply(weights_init)
    if optimizer == 'SGD':
        optim_d = optim.SGD(D.parameters(), lr=lrd)
        optim_g = optim.SGD(G.parameters(), lr=lrg)
    elif optimizer == 'Adam':
        optim_d = optim.Adam(D.parameters(), lr=lrd, betas=(0.5, 0.9))
        optim_g = optim.Adam(G.parameters(), lr=lrg, betas=(0.5, 0.9))
    elif optimizer == 'RMS':
        optim_d = optim.RMSprop(itertools.chain(D.parameters()), lr=lrd, alpha=0.9)
        optim_g = optim.RMSprop(itertools.chain(G.parameters()), lr=lrg, alpha=0.9)
        schedulers = []
        schedulers.append(get_scheduler(optim_d,2000))
        schedulers.append(get_scheduler(optim_g, 2000))



    criterionmse = nn.MSELoss()
    criterion = nn.BCELoss()
    zeros = torch.zeros(batch_size, device=device)
    ones = torch.ones(batch_size, device=device)

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))

    scale = 1
    if args is not None:
        scale = args.scale
    scale *= data.range
    old=False
    for iter in range(niter):
        if iter % 100 == 0:
            print(iter)

            noise_batch = noise.next_batch(512, device=device)
            fake_batch = G(noise_batch)
            fake_batch = fake_batch.data.cpu().numpy()

            real_batch = data.next_batch(512, device=device)
            real_batch2 = data.next_batch(512, device=device)

            plt.figure(fig.number)
            ax.clear()
            ax.scatter(real_batch[:, 0].cpu(), real_batch[:, 1].cpu(), s=2)
            ax.scatter(fake_batch[:, 0], fake_batch[:, 1], s=2, c='r', marker='+')
            ax.set_xlim((-scale, scale))
            ax.set_ylim((-scale, scale))

            coord, grad = visualize_grad_rel(G, D, criterion, fig, ax, scale=scale, device=device)
            plt.draw()
            plt.savefig(prefix + 'fig_%05d.pdf' % iter, bbox_inches='tight')
            plt.pause(0.1)

        # train D
        optim_d.zero_grad()


        noise_batch = noise.next_batch(batch_size, device=device)
        fake_batch = G(noise_batch)
        fake_batch = fake_batch.detach()


     #   fake_batch2 = fake_batch2.detach()
        predict_fake = D(fake_batch[:batch_size//2].detach(),fake_batch[batch_size//2:].detach())#ff 0
     #   print(predict_fake.shape)
        loss_fake = criterionmse.forward(predict_fake, zeros[:batch_size//2])
        predict_fake2 = D(real_batch2[:batch_size], real_batch[:batch_size])  # rr 0
        #   print(predict_fake.shape)
        loss_fake2 = criterionmse.forward(predict_fake2, zeros) ##FF 0


        real_batch = data.next_batch(batch_size, device=device)
        predict_real = D(real_batch,fake_batch)##fr <0 rf>0
        predict_real_ = D(fake_batch, real_batch)
        loss_real = -predict_real.mean()+predict_real_.mean()+\
                    0.01*( -predict_real.mean()+predict_real_.mean())*( -predict_real.mean()+predict_real_.mean())/torch.abs(real_batch-fake_batch).mean()/torch.abs(real_batch-fake_batch).mean()#- 0.02*criterionmse.forward((predict_real[:batch_size//2]+predict_real_[batch_size//2:]
                                                        #                                 -predict_fake2[:batch_size//2]-predict_fake),zeros[:batch_size//2])\

        if gp:
            gradpen = cal_gradpen_rel2(D, real_batch.detach(), fake_batch.detach(),
                                       center=center, LAMBDA=LAMBDA, alpha=alpha, device=device)
            print(float(gradpen))
            loss_d =0.5*loss_fake2 + 0.5*loss_fake + loss_real+ gradpen
        else:
            loss_d = loss_fake2 +loss_fake + loss_real
        loss_d.backward()
        optim_d.step()
        # for p in D.parameters():
        #        p.data.clamp_(-0.1,0.1)


        # train G

        optim_g.zero_grad()
        noise_batch = noise.next_batch(batch_size)
        noise_batch = noise_batch.to(device)
        noise_batch2 = noise.next_batch(batch_size, device=device)
        fake_batch2 = G(noise_batch2)
        fake_batch = G(noise_batch)
        if old:
            predict_fake_ = D(fake_batch,old_fake.detach())##ff >0
        else:
            fake_batch2 = G(noise_batch2)
            predict_fake_ = D(fake_batch, fake_batch2.detach())  ##ff >0
        predict_fake1 = D(fake_batch, real_batch)# rf=0
        predict_fake2 = D(real_batch,fake_batch2)# fr=0
        loss_g =-predict_fake_.mean()+0.5*criterionmse.forward(predict_fake1, zeros)+0.5*criterionmse.forward(predict_fake2, zeros)
       # loss_g =-predict_fake_.mean()+ nn.ReLU()(predict_fake2).mean()+ nn.ReLU()(-predict_fake1).mean()

        loss_g.backward()
        optim_g.step()
        old_fake = fake_batch.detach()
        old = True
        print('d',float(loss_d),'g:',float(loss_g))
       # for scheduler in schedulers:
        #    scheduler.step()

    return D, G

def GAN_LS_RE2(D, G, data, noise, niter=10000, batch_size=32, optimizer='Adam',
           lrg=1e-3, lrd=3e-3, center=0, LAMBDA=1, alpha=None, device='cuda', prefix='figs/', args=None,gp=False):
    D.to(device)
    G.to(device)
    #D.apply(weights_init)
    #G.apply(weights_init)
    if optimizer == 'SGD':
        optim_d = optim.SGD(D.parameters(), lr=lrd)
        optim_g = optim.SGD(G.parameters(), lr=lrg)
    elif optimizer == 'Adam':
        optim_d = optim.Adam(D.parameters(), lr=lrd, betas=(0.5, 0.9))
        optim_g = optim.Adam(G.parameters(), lr=lrg, betas=(0.5, 0.9))
    elif optimizer == 'RMS':
        optim_d = optim.RMSprop(itertools.chain(D.parameters()), lr=lrd, alpha=0.9)
        optim_g = optim.RMSprop(itertools.chain(G.parameters()), lr=lrg, alpha=0.9)
        schedulers = []
        schedulers.append(get_scheduler(optim_d,2000))
        schedulers.append(get_scheduler(optim_g, 2000))



    criterionmse = nn.MSELoss()
    criterion = nn.BCELoss()
    zeros = torch.zeros(batch_size, device=device)
    ones = torch.ones(batch_size, device=device)

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))

    scale = 1
    if args is not None:
        scale = args.scale
    scale *= data.range
    old=False
    for iter in range(niter):
        if iter % 100 == 0:
            print(iter)

            noise_batch = noise.next_batch(512, device=device)
            fake_batch = G(noise_batch)
            fake_batch = fake_batch.data.cpu().numpy()

            real_batch = data.next_batch(512, device=device)
            real_batch2 = data.next_batch(512, device=device)

            plt.figure(fig.number)
            ax.clear()
            ax.scatter(real_batch[:, 0].cpu(), real_batch[:, 1].cpu(), s=2)
            ax.scatter(fake_batch[:, 0], fake_batch[:, 1], s=2, c='r', marker='+')
            ax.set_xlim((-scale, scale))
            ax.set_ylim((-scale, scale))

            coord, grad = visualize_grad_rel(G, D, criterion, fig, ax, scale=scale, device=device)
            plt.draw()
            plt.savefig(prefix + 'fig_%05d.pdf' % iter, bbox_inches='tight')
            plt.pause(0.1)

        # train D
        optim_d.zero_grad()


        noise_batch = noise.next_batch(batch_size, device=device)
        fake_batch = G(noise_batch)
        fake_batch = fake_batch.detach()


     #   fake_batch2 = fake_batch2.detach()
        predict_fake = D(fake_batch[:batch_size//2].detach(),fake_batch[batch_size//2:].detach())#ff 0
     #   print(predict_fake.shape)
        loss_fake = criterionmse.forward(predict_fake, zeros[:batch_size//2])
        #predict_rr = D(real_batch2[:batch_size], real_batch[:batch_size])  # rr 0
        #   print(predict_fake.shape)
        #loss_fake2 = criterionmse.forward(predict_fake2, zeros) ##FF 0


        real_batch = data.next_batch(batch_size, device=device)
        predict_rf = D(real_batch,fake_batch)##fr -1 rf 1
        predict_fr = D(fake_batch, real_batch)
        loss_real = criterionmse.forward(predict_rf,ones)+criterionmse.forward(predict_fr,zeros-1)#+0.01*( -predict_real.mean()+predict_real_.mean())*( -predict_real.mean()+predict_real_.mean())/torch.abs(real_batch-fake_batch).mean()/torch.abs(real_batch-fake_batch).mean()#- 0.02*criterionmse.forward((predict_real[:batch_size//2]+predict_real_[batch_size//2:]
                                                        #                                 -predict_fake2[:batch_size//2]-predict_fake),zeros[:batch_size//2])\

        if gp:
            gradpen = cal_gradpen_rel2(D, real_batch.detach(), fake_batch.detach(),
                                       center=center, LAMBDA=LAMBDA, alpha=alpha, device=device)
            print(float(gradpen))
            loss_d = loss_fake + loss_real+ gradpen
        else:
            loss_d = loss_fake + loss_real
        loss_d.backward()
        optim_d.step()
        # for p in D.parameters():
        #        p.data.clamp_(-0.1,0.1)


        # train G

        optim_g.zero_grad()
        noise_batch = noise.next_batch(batch_size)
        noise_batch = noise_batch.to(device)
        noise_batch2 = noise.next_batch(batch_size, device=device)
        fake_batch2 = G(noise_batch2)
        fake_batch = G(noise_batch)
        if old:
            predict_fake_ = D(fake_batch,old_fake.detach())##ff 1
        else:
            fake_batch2 = G(noise_batch2)
            predict_fake_ = D(fake_batch, fake_batch2.detach())  ##ff 1
        predict_fake1 = D(fake_batch, real_batch)# rf 0
        predict_fake2 = D(real_batch,fake_batch2)# fr 0
        loss_g =criterionmse.forward(-predict_fake_,ones)+criterionmse.forward(predict_fake1, zeros)+criterionmse.forward(predict_fake2, zeros)
       # loss_g =-predict_fake_.mean()+ nn.ReLU()(predict_fake2).mean()+ nn.ReLU()(-predict_fake1).mean()

        loss_g.backward()
        optim_g.step()
        old_fake = fake_batch.detach()
        old = True
        print('d',float(loss_d),'g:',float(loss_g))
      #  for scheduler in schedulers:
       #     scheduler.step()

    return D, G


def ls_GAN(D, G, data, noise, niter=10000, batch_size=32, optimizer='Adam',
           lrg=1e-3, lrd=3e-3, center=0, LAMBDA=1, alpha=None, device='cuda', prefix='figs/', args=None):
    D.to(device)
    G.to(device)

    if optimizer == 'SGD':
        optim_d = optim.SGD(D.parameters(), lr=lrd)
        optim_g = optim.SGD(G.parameters(), lr=lrg)
    elif optimizer == 'Adam':
        optim_d = optim.Adam(D.parameters(), lr=lrd, betas=(0.5, 0.9))
        optim_g = optim.Adam(G.parameters(), lr=lrg, betas=(0.5, 0.9))
    elif optimizer == 'RMS':
        optim_d = optim.Adam(D.parameters(), lr=lrd, betas=(0.5, 0.9))
        optim_g = optim.Adam(G.parameters(), lr=lrg, betas=(0.5, 0.9))

    criterion = nn.MSELoss()

    zeros = torch.zeros(batch_size, device=device)
    ones = torch.ones(batch_size, device=device)

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))

    scale = 1
    if args is not None:
        scale = args.scale
    scale *= data.range

    for iter in range(niter):
        if iter % 100 == 0:
            print(iter)

            noise_batch = noise.next_batch(512, device=device)
            fake_batch = G(noise_batch)
            fake_batch = fake_batch.data.cpu().numpy()

            real_batch = data.next_batch(512, device=device)

            plt.figure(fig.number)
            ax.clear()
            ax.scatter(real_batch[:, 0].cpu(), real_batch[:, 1].cpu(), s=2)
            ax.scatter(fake_batch[:, 0], fake_batch[:, 1], s=2, c='r', marker='+')
            ax.set_xlim((-scale, scale))
            ax.set_ylim((-scale, scale))

            coord, grad = visualize_grad(G, D, criterion, fig, ax, scale=scale, device=device)
            plt.draw()
            plt.savefig(prefix + 'fig_%05d.pdf' % iter, bbox_inches='tight')
            plt.pause(0.1)

        # train D
        optim_d.zero_grad()
        real_batch = data.next_batch(batch_size, device=device)
        predict_real = D(real_batch)
        loss_real = criterion.forward(predict_real, ones)

        noise_batch = noise.next_batch(batch_size, device=device)
        fake_batch = G(noise_batch)
        fake_batch = fake_batch.detach()
        predict_fake = D(fake_batch)
        loss_fake = criterion.forward(predict_fake, zeros)
       ## gradpen = cal_gradpen(D, real_batch.detach(), fake_batch.detach(),
       #                       center=center, LAMBDA=LAMBDA, alpha=alpha, device=device)
        loss_d = loss_real + loss_fake #+ gradpen
        loss_d.backward()
        optim_d.step()

        # train G
        optim_g.zero_grad()
        noise_batch = noise.next_batch(batch_size)
        noise_batch = noise_batch.to(device)
        fake_batch = G(noise_batch)
        predict_fake = D(fake_batch)
        loss_g = criterion.forward(predict_fake, ones)
        loss_g.backward()
        optim_g.step()

    return D, G

def GAN_GP_PAC(D, G, data, noise, niter=10000, batch_size=32, optimizer='Adam',
           lrg=1e-3, lrd=3e-3, center=0, LAMBDA=1, alpha=None, device='cuda', prefix='figs/', args=None,gp=False):
    D.to(device)
    G.to(device)
    #D.apply(weights_init)
    #G.apply(weights_init)
    if optimizer == 'SGD':
        optim_d = optim.SGD(D.parameters(), lr=lrd)
        optim_g = optim.SGD(G.parameters(), lr=lrg)
    elif optimizer == 'Adam':
        optim_d = optim.Adam(D.parameters(), lr=lrd, betas=(0.5, 0.9))
        optim_g = optim.Adam(G.parameters(), lr=lrg, betas=(0.5, 0.9))
    elif optimizer == 'RMS':
        optim_d = optim.RMSprop(itertools.chain(D.parameters()), lr=lrd, alpha=0.9)
        optim_g = optim.RMSprop(itertools.chain(G.parameters()), lr=lrg, alpha=0.9)
        schedulers = []
        schedulers.append(get_scheduler(optim_d,2000))
        schedulers.append(get_scheduler(optim_g, 2000))



    criterionmse = nn.MSELoss()
    criterion = nn.BCELoss()
    zeros = torch.zeros(batch_size, device=device)
    ones = torch.ones(batch_size, device=device)

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))

    scale = 1
    if args is not None:
        scale = args.scale
    scale *= data.range
    old=False
    for iter in range(niter):
        if iter % 100 == 0:
            print(iter)

            noise_batch = noise.next_batch(512, device=device)
            fake_batch = G(noise_batch)
            fake_batch = fake_batch.data.cpu().numpy()

            real_batch = data.next_batch(512, device=device)
            real_batch2 = data.next_batch(512, device=device)

            plt.figure(fig.number)
            ax.clear()
            ax.scatter(real_batch[:, 0].cpu(), real_batch[:, 1].cpu(), s=2)
            ax.scatter(fake_batch[:, 0], fake_batch[:, 1], s=2, c='r', marker='+')
            ax.set_xlim((-scale, scale))
            ax.set_ylim((-scale, scale))

            coord, grad = visualize_grad_rel(G, D, criterion, fig, ax, scale=scale, device=device)
            plt.draw()
            plt.savefig(prefix + 'fig_%05d.pdf' % iter, bbox_inches='tight')
            plt.pause(0.1)

        # train D
        optim_d.zero_grad()


        noise_batch = noise.next_batch(batch_size, device=device)
        fake_batch = G(noise_batch)
        fake_batch = fake_batch.detach()


     #   fake_batch2 = fake_batch2.detach()
        predict_fake = D(fake_batch[:batch_size//2].detach(),fake_batch[batch_size//2:].detach())#ff 0
     #   print(predict_fake.shape)
        loss_fake = criterionmse.forward(predict_fake, zeros[:batch_size//2])
        predict_fake2 = D(real_batch2[:batch_size], real_batch[:batch_size])  # rr 1
        #   print(predict_fake.shape)
        loss_fake2 = criterionmse.forward(predict_fake2, ones)


      #  real_batch = data.next_batch(batch_size, device=device)
      #  predict_real = D(real_batch,fake_batch)##fr <0 rf>0
      #  predict_real_ = D(fake_batch, real_batch)
      #  loss_real = -predict_real.mean()+predict_real_.mean()+0.01*( -predict_real.mean()+predict_real_.mean())*( -predict_real.mean()+predict_real_.mean())/torch.abs(real_batch-fake_batch).mean()/torch.abs(real_batch-fake_batch).mean()#- 0.02*criterionmse.forward((predict_real[:batch_size//2]+predict_real_[batch_size//2:]
                                                        #                                 -predict_fake2[:batch_size//2]-predict_fake),zeros[:batch_size//2])\

        if gp:
            gradpen = cal_gradpen_rel2(D, real_batch.detach(), fake_batch.detach(),
                                       center=center, LAMBDA=LAMBDA, alpha=alpha, device=device)
            print(float(gradpen))
            loss_d =0.5*loss_fake2 + 0.5*loss_fake+ gradpen
        else:
            loss_d = loss_fake2 +loss_fake
        loss_d.backward()
        optim_d.step()
        # for p in D.parameters():
        #        p.data.clamp_(-0.1,0.1)


        # train G

        optim_g.zero_grad()
        noise_batch = noise.next_batch(batch_size)
        noise_batch = noise_batch.to(device)
        noise_batch2 = noise.next_batch(batch_size, device=device)
        fake_batch2 = G(noise_batch2)
        fake_batch = G(noise_batch)
        if old:
            predict_fake_ = D(fake_batch,old_fake.detach())##ff >0
        else:
            fake_batch2 = G(noise_batch2)
            predict_fake_ = D(fake_batch, fake_batch2.detach())  ##ff >0

        loss_g =0.5*criterionmse.forward(predict_fake_, ones)
       # loss_g =-predict_fake_.mean()+ nn.ReLU()(predict_fake2).mean()+ nn.ReLU()(-predict_fake1).mean()

        loss_g.backward()
        optim_g.step()
        old_fake = fake_batch.detach()
        old = True
        print('d',float(loss_d),'g:',float(loss_g))
    for scheduler in schedulers:
        scheduler.step()

    return D, G



def GAN_REL_orig(D, G, data, noise, niter=10000, batch_size=32, optimizer='Adam',
           lrg=1e-3, lrd=3e-3, center=0, LAMBDA=1, alpha=None, device='cuda', prefix='figs/', args=None,gp=False):
    D.to(device)
    G.to(device)
    #D.apply(weights_init)
    #G.apply(weights_init)
    if optimizer == 'SGD':
        optim_d = optim.SGD(D.parameters(), lr=lrd)
        optim_g = optim.SGD(G.parameters(), lr=lrg)
    elif optimizer == 'Adam':
        optim_d = optim.Adam(D.parameters(), lr=lrd, betas=(0.5, 0.9))
        optim_g = optim.Adam(G.parameters(), lr=lrg, betas=(0.5, 0.9))
    elif optimizer == 'RMS':
        optim_d = optim.RMSprop(itertools.chain(D.parameters()), lr=lrd, alpha=0.9)
        optim_g = optim.RMSprop(itertools.chain(G.parameters()), lr=lrg, alpha=0.9)
        schedulers = []
        schedulers.append(get_scheduler(optim_d,2000))
        schedulers.append(get_scheduler(optim_g, 2000))



    criterionmse = nn.MSELoss()
    criterion = nn.BCELoss()
    zeros = torch.zeros(batch_size, device=device)
    ones = torch.ones(batch_size, device=device)

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))

    scale = 1
    if args is not None:
        scale = args.scale
    scale *= data.range
    old=False
    for iter in range(niter):
        if iter % 100 == 0:
            print(iter)

            noise_batch = noise.next_batch(512, device=device)
            fake_batch = G(noise_batch)
            fake_batch = fake_batch.data.cpu().numpy()

            real_batch = data.next_batch(512, device=device)
            real_batch2 = data.next_batch(512, device=device)

            plt.figure(fig.number)
            ax.clear()
            ax.scatter(real_batch[:, 0].cpu(), real_batch[:, 1].cpu(), s=2)
            ax.scatter(fake_batch[:, 0], fake_batch[:, 1], s=2, c='r', marker='+')
            ax.set_xlim((-scale, scale))
            ax.set_ylim((-scale, scale))

            coord, grad = visualize_grad_rel(G, D, criterion, fig, ax, scale=scale, device=device)
            plt.draw()
            plt.savefig(prefix + 'fig_%05d.pdf' % iter, bbox_inches='tight')
            plt.pause(0.1)

        # train D
        optim_d.zero_grad()


        noise_batch = noise.next_batch(batch_size, device=device)
        fake_batch = G(noise_batch)
        fake_batch = fake_batch.detach()


     #   fake_batch2 = fake_batch2.detach()
        predict_ff = D(fake_batch[:batch_size//2].detach(),fake_batch[batch_size//2:].detach())#ff -1
     #   print(predict_fake.shape)

        predict_rr = D(real_batch2[:batch_size], real_batch[:batch_size])  # rr 1
        #   print(predict_fake.shape)



        real_batch = data.next_batch(batch_size, device=device)
        predict_rf = D(real_batch,fake_batch)##fr =0

       # loss_real =  criterionmse.forward(predict_real_,zeros[:batch_size])+criterionmse.forward(predict_real,zeros[:batch_size])

        loss_d =predict_rr.mean()+10000*nn.ReLU()(predict_rr-predict_rf+100).mean()+\
                predict_ff.mean()+10000*nn.ReLU()(predict_ff[:predict_rf.shape[0]]-predict_rf[:predict_ff.shape[0]]+100).mean()
        loss_d.backward()
        optim_d.step()
        # for p in D.parameters():
        #        p.data.clamp_(-0.1,0.1)


        # train G

        optim_g.zero_grad()
        noise_batch = noise.next_batch(batch_size)
        noise_batch = noise_batch.to(device)
        noise_batch2 = noise.next_batch(batch_size, device=device)
        fake_batch2 = G(noise_batch2)
        fake_batch = G(noise_batch)
        if old:
            predict_fake_ = D(fake_batch,old_fake.detach())##ff >0
        else:
            fake_batch2 = G(noise_batch2)
            predict_fake_ = D(fake_batch, fake_batch2.detach())  ##ff >0
        if np.random.rand()>0.5:
            predict_fr = D(fake_batch2, real_batch)  ##fr >1
        else:
            predict_fr = D(real_batch,fake_batch2)  ##rf >1
        loss_g =predict_fr.mean()-predict_fake_.mean()
       # loss_g =-predict_fake_.mean()+ nn.ReLU()(predict_fake2).mean()+ nn.ReLU()(-predict_fake1).mean()

        loss_g.backward()
        optim_g.step()
        old_fake = fake_batch.detach()
        old = True
        print('d',float(loss_d),'g:',float(loss_g))
      #  for scheduler in schedulers:
       #     scheduler.step()

    return D, G



def WGAN_GP(D, G, data, noise, niter=10000, ncritic=5, batch_size=32, optimizer='Adam',
                lrg=1e-3, lrd=3e-3, center=0, LAMBDA=1, alpha=None, device='cuda', prefix='figs/',gp=False, args=None):
    # D.apply(weights_init)
    # G.apply(weights_init)
    D.to(device)
    G.to(device)
    if optimizer == 'SGD':
        optim_d = optim.SGD(D.parameters(), lr=lrd)
        optim_g = optim.SGD(G.parameters(), lr=lrg)
    elif optimizer == 'Adam':
        optim_d = optim.Adam(D.parameters(), lr=lrd, betas=(0.5, 0.9))
        optim_g = optim.Adam(G.parameters(), lr=lrg, betas=(0.5, 0.9))

    criterion = nn.BCELoss()
    scale = 1

    if args is not None:
        scale = args.scale
    scale *= data.range
    print('scale', scale, data.range)

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))

    for iter in range(niter):
        if iter % 1000 == 0:
            print(iter)
            D.zero_grad()
            G.zero_grad()

            noise_batch = noise.next_batch(512, device=device)
            fake_batch = G(noise_batch)
            fake_batch = fake_batch.data.cpu().numpy()

            real_batch = data.next_batch(512, device=device)

            ax.clear()
            ax.scatter(real_batch[:, 0].cpu(), real_batch[:, 1].cpu(), s=2)
            ax.scatter(fake_batch[:, 0], fake_batch[:, 1], s=2, c='r', marker='+')
            ax.set_xlim((-scale, scale))
            ax.set_ylim((-scale, scale))

            visualize_grad(G, D, criterion, fig, ax, scale=scale, device=device)
            plt.draw()
            plt.savefig(prefix + 'fig_%05d.pdf' % iter, bbox_inches='tight')
            plt.pause(0.1)

        # train D
        D.zero_grad()
        for p in D.parameters():
            p.requires_grad_(True)
        for i in range(ncritic):
            optim_d.zero_grad()
            real_batch = data.next_batch(batch_size)
            real_batch = real_batch.to(device)
            loss_real = D(real_batch).mean()

            noise_batch = noise.next_batch(batch_size)
            noise_batch = noise_batch.to(device)
            fake_batch = G(noise_batch)
            fake_batch = fake_batch.detach()
            loss_fake = D(fake_batch).mean()
            if gp:
                gradpen = cal_gradpen(D, real_batch.data, fake_batch.data,
                                  center=center, LAMBDA=LAMBDA, alpha=alpha, device=device)

                loss_d = -loss_real + loss_fake + gradpen
            else:
                loss_d = -loss_real+loss_fake
            loss_d.backward()
            optim_d.step()

        # train G
        G.zero_grad()
        optim_g.zero_grad()
        for p in D.parameters():
            p.requires_grad_(False)

        noise_batch = noise.next_batch(batch_size)
        noise_batch = noise_batch.to(device)
        fake_batch = G(noise_batch)
        loss_g = -D(fake_batch).mean()
        loss_g.backward()
        optim_g.step()
        print('d',float(loss_d),'g:',float(loss_g))

    return D, G


def cal_gradpen(netD, real_data, fake_data, center=0, alpha=None, LAMBDA=1, device=None):
    if alpha is not None:
        alpha = torch.tensor(alpha, device=device)  # torch.rand(real_data.size(0), 1, device=device)
    else:
        alpha = torch.rand(real_data.size(0), 1, device=device)
    alpha = alpha.expand(real_data.size())
    size = min(fake_data.shape[0],real_data.shape[0])
    print(size,real_data.shape,fake_data.shape)
    interpolates = alpha[:size] * real_data[:size,:] + ((1 - alpha)[:size] * fake_data[:size,:])

    interpolates.requires_grad_(True)

    disc_interpolates = netD(interpolates)

    gradients = ag.grad(outputs=disc_interpolates, inputs=interpolates,
                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                        create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - center) ** 2).mean() * LAMBDA
    return gradient_penalty

def cal_gradpen_rel2(netD, real_data, fake_data, center=0, alpha=None, LAMBDA=1, device=None):
    if alpha is not None:
        alpha = torch.tensor(alpha, device=device)  # torch.rand(real_data.size(0), 1, device=device)
    else:
        alpha = torch.rand(real_data.size(0), 1, device=device)
    alpha = alpha.expand(real_data.size())
    size = min(fake_data.shape[0],real_data.shape[0])
  #  print(size,real_data.shape,fake_data.shape)
    interpolates = alpha[:size] * real_data[:size,:] + ((1 - alpha)[:size] * fake_data[:size,:])

    interpolates.requires_grad_(True)

    disc_interpolates = netD(interpolates,fake_data)

    gradients = ag.grad(outputs=disc_interpolates, inputs=interpolates,
                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                        create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - center) ** 2).mean() * LAMBDA
    return gradient_penalty

def get_scheduler(optimizer,lr_decay_iters):

    scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_decay_iters, gamma=0.1)

    return scheduler
def visualize_grad_rel(G: Generator, D: Criticr_realtion2, criterion, fig, ax, scale, device=None):
    nticks = 20
    noise_batch = (torch.rand(nticks * nticks, 2, device=device) - 0.5) * 4
    noise_batch2 = (torch.rand(nticks * nticks, 2, device=device) - 0.5) * 4
    ones = torch.ones(nticks * nticks, 1, device=device)

    step = 2 * scale / nticks
    for i in range(nticks):
        for j in range(nticks):
            noise_batch[i * nticks + j, 0] = -scale + i * step
            noise_batch[i * nticks + j, 1] = -scale + j * step

    noise_batch.requires_grad_()
    with torch.enable_grad():
        out_batch = D(noise_batch,noise_batch2.detach())

        loss = -out_batch.mean()
        loss.backward()

    coord = noise_batch.data.cpu().numpy()
    grad = -noise_batch.grad.cpu().numpy()

    ax.quiver(coord[:, 0], coord[:, 1], grad[:, 0], grad[:, 1])
    return coord, grad

def visualize_grad_rel3(G: Generator, D: Criticr_realtion2, criterion, fig, ax, scale, device=None):
    nticks = 20
    noise_batch = (torch.rand(nticks * nticks, 2, device=device) - 0.5) * 4
    noise_batch2 = (torch.rand(nticks * nticks, 2, device=device) - 0.5) * 4
    noise_batch3 = (torch.rand(nticks * nticks, 2, device=device) - 0.5) * 4
    ones = torch.ones(nticks * nticks, 1, device=device)

    step = 2 * scale / nticks
    for i in range(nticks):
        for j in range(nticks):
            noise_batch[i * nticks + j, 0] = -scale + i * step
            noise_batch[i * nticks + j, 1] = -scale + j * step

    noise_batch.requires_grad_()
    with torch.enable_grad():
        out_batch = D(noise_batch,noise_batch2.detach(),noise_batch3.detach())

        loss = -out_batch.mean()
        loss.backward()

    coord = noise_batch.data.cpu().numpy()
    grad = -noise_batch.grad.cpu().numpy()

    ax.quiver(coord[:, 0], coord[:, 1], grad[:, 0], grad[:, 1])
    return coord, grad
def visualize_grad(G: Generator, D: Critic, criterion, fig, ax, scale, device=None):
    nticks = 20
    noise_batch = (torch.rand(nticks * nticks, 2, device=device) - 0.5) * 4
    noise_batch2 = (torch.rand(nticks * nticks, 2, device=device) - 0.5) * 4
    ones = torch.ones(nticks * nticks, 1, device=device)

    step = 2 * scale / nticks
    for i in range(nticks):
        for j in range(nticks):
            noise_batch[i * nticks + j, 0] = -scale + i * step
            noise_batch[i * nticks + j, 1] = -scale + j * step

    noise_batch.requires_grad_()
    with torch.enable_grad():
        out_batch = D(noise_batch)


        loss = -out_batch.mean()
        loss.backward()

    coord = noise_batch.data.cpu().numpy()
    grad = -noise_batch.grad.cpu().numpy()

    ax.quiver(coord[:, 0], coord[:, 1], grad[:, 0], grad[:, 1])
    return coord, grad
def show_grad(coord, grad, fig, ax):
    ax.quiver(coord[:, 0], coord[:, 1], grad[:, 0], grad[:, 1])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--beta', type=float, default=1)
    parser.add_argument('--nhidden', type=int, default=64, help='number of hidden neurons')
    parser.add_argument('--gnlayers', type=int, default=2, help='number of hidden layers in generator')
    parser.add_argument('--dnlayers', type=int, default=4, help='number of hidden layers in discriminator/critic')
    parser.add_argument('--niters', type=int, default=40000, help='number of iterations')
    parser.add_argument('--device', type=str, default='cuda', help='id of the gpu. -1 for cpu')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--center', type=float, default=1., help='gradpen center')
    parser.add_argument('--LAMBDA', type=float, default=10., help='gradpen weight')
    parser.add_argument('--alpha', type=float, default=None, help='interpolation weight between reals and fakes')
    parser.add_argument('--lrg', type=float, default=2e-3, help='lr for G')
    parser.add_argument('--lrd', type=float, default=1e-3, help='lr for D')
    parser.add_argument('--dataset', type=str, default='swissroll',help='dataset to use: 8Gaussians|25Gaussians | swissroll')
    parser.add_argument('--scale', type=float, default=10., help='data scaling')
    parser.add_argument('--loss', type=str, default='relorig', help='gan|wgan|relgan|relativistic|lsgan')
    parser.add_argument('--optim', type=str, default='RMS', help='optimizer to use')
    parser.add_argument('--ncritic', type=int, default=1, help='critic iters / generator iter')
    parser.add_argument('--gp', type=str, default=False, help='gp')
    args = parser.parse_args()
    if args.gp:
        prefix = 'figs/gp/%s_%s_gradfield_center_%.2f_alpha_%s_lambda_%.2f_lrg_%.5f_lrd_%.5f_nhidden_%d_scale_%.2f' \
                 '_optim_%s_gnlayers_%d_dnlayers_%d_ncritic_%d_beta_%d/' % \
                 (args.loss, args.dataset, args.center, str(args.alpha), args.LAMBDA, args.lrg, args.lrd, args.nhidden,
                  args.scale, args.optim, args.gnlayers, args.dnlayers,
                  args.ncritic,args.beta)
    else:
        prefix = 'figs/no_gp/%s_%s_gradfield_lrg_%.5f_lrd_%.5f_nhidden_%d_scale_%.2f' \
                 '_optim_%s_gnlayers_%d_dnlayers_%d_ncritic_%d_beta_%d/' % \
                 (args.loss, args.dataset, args.lrg, args.lrd, args.nhidden,
                  args.scale, args.optim, args.gnlayers, args.dnlayers,
                  args.ncritic,args.beta)

    print(prefix)
    if not os.path.exists('figs'):
        os.mkdir('figs')
    if not os.path.exists(prefix):
        os.mkdir(prefix)

    G = Generator(args.nhidden, args.gnlayers)
    if args.loss == 'gan':
        D = Critic(args.nhidden, args.dnlayers)
    elif args.loss=='wgan':
        D = Critic(args.nhidden, args.dnlayers)
    elif args.loss == 'pacgan':
        D = Criticr_pac2(args.nhidden, args.dnlayers)
    elif args.loss == 'relorig':
        D = Criticr_pac2(args.nhidden,args.dnlayers)
    else:
        D = Critic(args.nhidden, args.dnlayers)

    noise = NoiseDataset()
    data = ToyDataset(distr=args.dataset, scale=args.scale)
    if args.loss == 'gan':
        print(args.loss)
        GAN_GP(D, G, data, noise, niter=args.niters+1, batch_size=args.batch_size, optimizer=args.optim,
               lrg=args.lrg, lrd=args.lrd, center=args.center, LAMBDA=args.LAMBDA, alpha=args.alpha,
               device=args.device, prefix=prefix, args=args,gp = args.gp)
    elif args.loss == 'wgan':
        print(args.loss)
        WGAN_GP(D, G, data, noise, niter=args.niters + 1, ncritic=args.ncritic, batch_size=args.batch_size,
                optimizer=args.optim, lrg=args.lrg, lrd=args.lrd, center=args.center, LAMBDA=args.LAMBDA,
                alpha=args.alpha, device=args.device, prefix=prefix, args=args,gp = args.gp)
    elif args.loss == 'lsgan':
        print(args.loss)
        ls_GAN(D, G, data, noise, niter=args.niters + 1, batch_size=args.batch_size,
               optimizer=args.optim, lrg=args.lrg, lrd=args.lrd, center=args.center, LAMBDA=args.LAMBDA,
               alpha=args.alpha, device=args.device, prefix=prefix, args=args)

    elif args.loss == 'relorig':
        print(args.loss)
        GAN_REL_orig(D, G, data, noise, niter=args.niters + 1, batch_size=args.batch_size, optimizer=args.optim,
                   lrg=args.lrg, lrd=args.lrd, center=args.center, LAMBDA=args.LAMBDA, alpha=args.alpha,
                   device=args.device, prefix=prefix, args=args, gp=args.gp)
    elif args.loss == 'pacgan':
        print(args.loss)
        GAN_GP_PAC(D, G, data, noise, niter=args.niters + 1, batch_size=args.batch_size, optimizer=args.optim,
                   lrg=args.lrg, lrd=args.lrd, center=args.center, LAMBDA=args.LAMBDA, alpha=args.alpha,
                   device=args.device, prefix=prefix, args=args, gp=args.gp)

