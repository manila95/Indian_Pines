-- Reading data from files

matio = require 'matio'
require 'nn'
require 'torch'


--  --------------------------- Reading In the Training Data --------------------------

print("Loading Data ")

train_patches = matio.load("../data/train/0.mat","train_patch")
train_labels = matio.load("../data/train/0.mat","train_labels")

for i = 1, 30 do
	file_name = "../data/train/" .. tostring(i) .. ".mat"
	patch = matio.load(file_name,"train_patch")
	label = matio.load(file_name,"train_labels")
	train_patches = torch.cat(train_patches,patch,1)
	train_labels = torch.cat(train_labels,label,1)
	print(i)
end

print("Training Data Loaded")

test_patches = matio.load("../data/test/0.mat","test_patch")
test_labels = matio.load("../data/test/0.mat","test_labels")

for i = 1, 2 do
	file_name = "../data/test/" .. tostring(i) .. ".mat"
	patch = matio.load(file_name,"test_patch")
	label = matio.load(file_name,"test_labels")
	test_patches = torch.cat(test_patches,patch,1)
	test_labels = torch.cat(test_labels,label,1)
end

print("Test Data Loaded")



