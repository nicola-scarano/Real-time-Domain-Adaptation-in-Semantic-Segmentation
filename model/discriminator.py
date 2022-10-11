# import torch
# from torch import nn
#
# #model based on paper 6 and https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
# class Discriminator(nn.Module):
#     def __init__(self, num_classes):
#         super(Discriminator, self).__init__()
#         self.main = nn.Sequential(
#             # layer 1
#             nn.Conv2d(num_classes, 64, (4*4), 2, bias=False),
#             nn.LeakyReLU(0.2, inplace=True),
#             # layer 2
#             nn.Conv2d(64, 128, (4*4), 2, bias=False),
#             nn.LeakyReLU(0.2, inplace=True),
#             # layer 3
#             nn.Conv2d(128 , 256, (4*4), 2, bias=False),
#             nn.LeakyReLU(0.2, inplace=True),
#             # layer 4
#             nn.Conv2d(256, 512, (4*4), 2, bias=False),
#             nn.LeakyReLU(0.2, inplace=True),
#             # layer 5
#             nn.Conv2d(512, 1, (4*4), 2, bias=False),
#             nn.Upsample(scale_factor=32, mode='bilinear') #https://pytorch.org/docs/stable/generated/torch.nn.Upsample.html
#         )
#
#     def forward(self, input):
#         return self.main(input)

import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):

	def __init__(self, num_classes, ndf = 64):
		super(Discriminator, self).__init__()

		self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1)
		self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
		self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
		self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)
		self.classifier = nn.Conv2d(ndf*8, 1, kernel_size=4, stride=2, padding=1)

		self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
		#self.up_sample = nn.Upsample(scale_factor=32, mode='bilinear')
		#self.sigmoid = nn.Sigmoid()


	def forward(self, x):
		x = self.conv1(x)
		x = self.leaky_relu(x)
		x = self.conv2(x)
		x = self.leaky_relu(x)
		x = self.conv3(x)
		x = self.leaky_relu(x)
		x = self.conv4(x)
		x = self.leaky_relu(x)
		x = self.classifier(x)
		#x = self.up_sample(x)
		#x = self.sigmoid(x)

		return x