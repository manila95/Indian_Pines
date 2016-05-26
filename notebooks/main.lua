-- Reading data from files

matio = require 'matio'
require 'nn'
require 'torch'
require 'optim'
require 'image'



classes = {'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16'}

geometry = {28,28}

local opt = lapp[[
	-tr,--trainsize     (default 6000)       number of patches in the training data
	-te,--testsize      (default 1000)       number of patches in the test data
	-p,--plot                                plot while training the network 
	-r,--learningRate   (default 0.05)        learning rate
	-b,--batchSize      (default 10)          batch size
	-m,--momentum       (default 0)           momentum
	--coefL1            (default 0)           L1 penalty on the weights
	--coefL2            (default 0)           L2 penalty on the weights
	-th,--threads       (default 4)           number of threads
]]

torch.manualSeed(1) -- fixing the seed

torch.setdefaulttensortype('torch.FloatTensor')


--  --------------------------- Reading In the Training Data --------------------------



function load_train()

	tr = opt.tr/1000 - 1

	train_patches = matio.load("../data/train/0.mat","train_patch")
	train_labels = matio.load("../data/train/0.mat","train_labels")

	if th ~= 0 then
		for i = 1, tr do
			file_name = "../data/train/" .. tostring(i) .. ".mat"
			patch = matio.load(file_name,"train_patch")
			label = matio.load(file_name,"train_labels")
			train_patches = torch.cat(train_patches,patch,1)
			train_labels = torch.cat(train_labels,label,1)
		end
	end

	print("Training Data Loaded")

	trainset = {}
	trainset.data = train_patches
	trainset.labels = train_labels

	local labelvector = torch.zeros(10)

    setmetatable(trainset, {__index = function(self, index)
			     local input = self.data[index]
			     local class = self.labels[index]
			     local label = labelvector:zero()
			     label[class] = 1
			     local example = {input, label}
                                       return example
    end})

    return trainset
end

function load_test()

	te = opt.te/1000 - 1

	test_patches = matio.load("../data/test/0.mat","test_patch")
	test_labels = matio.load("../data/test/0.mat","test_labels")

	if te ~= 0 then
		for i = 1, te do
			file_name = "../data/test/" .. tostring(i) .. ".mat"
			patch = matio.load(file_name,"test_patch")
			label = matio.load(file_name,"test_labels")
			test_patches = torch.cat(test_patches,patch,1)
			test_labels = torch.cat(test_labels,label,1)
		end
	end

	print("Test Data Loaded")

	testset = {}
	testset.data = test_patches
	testset.labels = test_labels

	local labelvector = torch.zeros(10)

    setmetatable(testset, {__index = function(self, index)
			     local input = self.data[index]
			     local class = self.labels[index]
			     local label = labelvector:zero()
			     label[class] = 1
			     local example = {input, label}
                                       return example
   	end})

    return testset
end



-- --------------------------- Defining The Convolutional Neural Network (leNet) -----------------------


model = nn.Sequential()
model:add(nn.SpatialConvolution(220,500,5,5))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(2,2,2,2))
model:add(nn.SpatialConvolution(6,100,5,5))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(2,2,2,2))
model:add(nn.View(100*4*4))
model:add(nn.Linear(16*4*4, 200))
model:add(nn.ReLU())
model:add(nn.Linear(120, 84))
model:add(nn.ReLU())
model:add(nn.Linear(84, 16))
model:add(nn.LogSoftMax())

-- ---------------------------        Defining The Loss Function      ------------------------------

criterion = nn.ClassNLLCriterion() -- negative log likelihood criterion 


-- ---------------------------        Training The Network        -----------------------------------

-- retrieve parameters and gradients
parameters,gradParameters = model:getParameters()


confusion = optim.ConfusionMatrix(classes) -- confusion across classes

-- log results to files
trainLogger = optim.Logger(paths.concat("../logs/", 'train.log'))
testLogger = optim.Logger(paths.concat("../logs/", 'test.log'))

function train(dataset)

	epoch = epoch or 1 -- tracking epoch

	local time = sys.clock() -- tracking time 

	print('<trainer> on training set:')
	print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')



end


trainer = nn.StochasticGradient(net, criterion)
trainer.learningRate = 0.001
trainer.maxIteration = 10

trainer:train(trainset)

-- ----------------------------        Testing The Network        -----------------------------------


correct = 0
for i=1,10000 do
    local groundtruth = testset.label[i]
    local prediction = net:forward(testset.data[i])
    local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
    if groundtruth == indices[1] then
        correct = correct + 1
    end
end

print(correct, 100*correct/10000 .. ' % ')








