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

	for i = 1, train_patches:size(1),opt.batchSize do
		-- creating mini batch
		local inputs = torch.Tensor(opt.batchSize,220, 28, 28)
		local targets = torch.Tensor(opt.batchSize)
		local j = 1
		for k = t, math.min(t+opt.batchSize-1, train_patches:size(1)) do
			   -- load new sample
            local sample = dataset[i]
			local input = sample[1]:clone()
			local _,target = sample[2]:clone():max(1)
			target = target:squeeze()
			inputs[j] = input
			targets[j] = target
			j = j + 1
		end

		-- create closure to evaluate f(X) and df/dX
		local feval = function(x)
			 -- just in case:
			collectgarbage()

			 -- get new parameters
			if x ~= parameters then
			parameters:copy(x)
			end

			 -- reset gradients
			gradParameters:zero()

			 -- evaluate function for complete mini batch
			local outputs = model:forward(inputs)
			local f = criterion:forward(outputs, targets)

			 -- estimate df/dW
			local df_do = criterion:backward(outputs, targets)
			model:backward(inputs, df_do)

			-- penalties (L1 and L2):
			if opt.coefL1 ~= 0 or opt.coefL2 ~= 0 then
			-- locals:
			local norm,sign= torch.norm,torch.sign

			-- Loss:
			f = f + opt.coefL1 * norm(parameters,1)
			f = f + opt.coefL2 * norm(parameters,2)^2/2

			-- Gradients:
			gradParameters:add( sign(parameters):mul(opt.coefL1) + parameters:clone():mul(opt.coefL2) )
			end

			-- update confusion
			for i = 1,opt.batchSize do
			confusion:add(outputs[i], targets[i])
			end

			-- return f and df/dX
			return f,gradParameters
		end

		-- Perform SGD step:
		sgdState = sgdState or {
		learningRate = opt.learningRate,
		momentum = opt.momentum,
		learningRateDecay = 5e-7
		}
		optim.sgd(feval, parameters, sgdState)

		-- disp progress
		xlua.progress(t, test_patches:size(1))

	-- time taken
	time = sys.clock() - time
	time = time / dataset:size()
	print("<trainer> time to learn 1 sample = " .. (time*1000) .. 'ms')

	-- print confusion matrix
	print(confusion)
	trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
	confusion:zero()

	-- save/log current net
	local filename = paths.concat(opt.save, 'mnist.net')
	os.execute('mkdir -p ' .. sys.dirname(filename))
	if paths.filep(filename) then
		os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
	end
	print('<trainer> saving network to '..filename)
	-- torch.save(filename, model)

	-- next epoch
	epoch = epoch + 1

end

-- ----------------------------        Testing The Network        -----------------------------------

-- test function
function test(dataset)
   -- local vars
   local time = sys.clock()

   -- test over given dataset
   print('<trainer> on testing Set:')
   for t = 1,dataset:size(),opt.batchSize do
      -- disp progress
      xlua.progress(t, dataset:size())

      -- create mini batch
      local inputs = torch.Tensor(opt.batchSize,1,geometry[1],geometry[2])
      local targets = torch.Tensor(opt.batchSize)
      local k = 1
      for i = t,math.min(t+opt.batchSize-1,dataset:size()) do
         -- load new sample
         local sample = dataset[i]
         local input = sample[1]:clone()
         local _,target = sample[2]:clone():max(1)
         target = target:squeeze()
         inputs[k] = input
         targets[k] = target
         k = k + 1
      end

      -- test samples
      local preds = model:forward(inputs)

      -- confusion:
      for i = 1,opt.batchSize do
         confusion:add(preds[i], targets[i])
      end
   end

   -- timing
   time = sys.clock() - time
   time = time / dataset:size()
   print("<trainer> time to test 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)
   testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
   confusion:zero()
end








