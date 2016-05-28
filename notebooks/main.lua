require 'torch'
require 'nn'
require 'optim'
require 'image'
require 'load_pines'
require 'pl'
require 'paths'


------------------------------------------- Setting the Command Line Options --------------------------------------------------

local opt = lapp[[
   --train            (default 31000)        train size
   --test             (default 4000)        test size
   -r,--learningRate  (default 0.05)        learning rate
   -b,--batchSize     (default 10)          batch size
   -m,--momentum      (default 0)           momentum
   --epoch            (default 5)           max number of epochs
   --coefL1           (default 0)           L1 penalty on the weights
   --coefL2           (default 0)           L2 penalty on the weights
   -t,--threads       (default 4)           number of threads
]]

-- fixing seed

torch.manualSeed(1)

-- threads
torch.setnumthreads(opt.threads)

print('===> Setting number of threads to ' .. torch.getnumthreads())

torch.setdefaulttensortype('torch.FloatTensor')


-- ------------------------------------------- Defining the classes --------------------------------------------------------

classes = {'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16'}


--------------------------------------   Defining the Convolutional Neural Network   ----------------------------------------


model = nn.Sequential()
model:add(nn.SpatialConvolution(220,500,5,5))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(2,2,2,2))
model:add(nn.SpatialConvolution(500,100,5,5))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(2,2,2,2))
model:add(nn.View(100*4*4))
model:add(nn.Linear(100*4*4, 200))
model:add(nn.ReLU())
model:add(nn.Linear(200, 84))
model:add(nn.ReLU())
model:add(nn.Linear(84, 16))
model:add(nn.LogSoftMax())

-- Retrieve parameters and gradients
parameters,gradParameters = model:getParameters()


print('===> Using model:')
print(model)

-------------------------------------------------------- Defining the Loss Function --------------------------------------------------

criterion = nn.ClassNLLCriterion()

-------------------------------------------------------- Loading the train and test data ---------------------------------------------


if opt.train > 30000 then
   local maxLoad = 31
   trainData = loadTrainSet(maxLoad)
else 
   local maxLoad = opt.train / 1000
   trainData = loadTrainSet(maxLoad)
end

if opt.test > 3000 then
   local maxLoad = 3
   testData = loadTestSet(maxLoad)
else
   local maxLoad = opt.test / 1000
   testData = loadTestSet(maxLoad)
end


------------------------------------------------------- Defining The Training And Testing Function ------------------------------------

-- Recording the confusion across classes

confusion = optim.ConfusionMatrix(classes)

-- log results to files
trainLogger = optim.Logger(paths.concat("../logs/", 'train.log'))
testLogger = optim.Logger(paths.concat("../logs/", 'test.log'))


function train(dataset)
   -- Tracking epoch
   epoch = epoch or 1

   local time = sys.clock() -- tracking time

   -- do one epoch
   print('===> Learning on training set:')
   print("===> Epoch Number :  " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
   for i = 1,dataset:size(),opt.batchSize do
      -- creating mini batch
      local inputs = torch.Tensor(opt.batchSize,220, 28, 28)
      local targets = torch.Tensor(opt.batchSize)
      local k = 1
      for j = i,math.min(i+opt.batchSize-1,dataset:size()) do
         -- load new sample
         local sample = dataset[j]
         local input = sample[1]:clone()
         local _,target = sample[2]:clone():max(1)
         target = target:squeeze()
         inputs[k] = input
         targets[k] = target
         k = k + 1
      end

      -- Function Closure for calculating f(X) and df/dX
      local feval = function(x)

         collectgarbage() -- if there is any

         -- getting new parameters
         if x ~= parameters then
            parameters:copy(x)
         end

         -- resetting gradients to zero
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

         -- updating confusion matrix
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
      xlua.progress(i, dataset:size())
      
   end
   
   -- time taken
   time = sys.clock() - time
   time = time / dataset:size()
   print("===> time to learn 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)
   trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
   confusion:zero()

   -- save/log current net
   local filename = paths.concat("../logs/", 'cnn.net')
   os.execute('mkdir -p ' .. sys.dirname(filename))
   if paths.filep(filename) then
      os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
   end
   print('===> saving network to '..filename)

   -- next epoch
   epoch = epoch + 1
end

-- test function

function test(dataset)

   local time = sys.clock()

   print('===> Predicting on testing Set:')
   for i = 1,dataset:size(),opt.batchSize do
      -- display progress
      xlua.progress(i, dataset:size())

      -- creating mini batch
      local inputs = torch.Tensor(opt.batchSize,220, 28, 28)
      local targets = torch.Tensor(opt.batchSize)
      local k = 1
      for j = i,math.min(i+opt.batchSize-1,dataset:size()) do
         -- load new sample
         local sample = dataset[j]
         local input = sample[1]:clone()
         local _,target = sample[2]:clone():max(1)
         target = target:squeeze()
         inputs[k] = input
         targets[k] = target
         k = k + 1
      end

      -- Making Predictions
      local preds = model:forward(inputs)

      -- confusion matrix:
      for i = 1,opt.batchSize do
         confusion:add(preds[i], targets[i])
      end
   end

   -- timing
   time = sys.clock() - time
   time = time / dataset:size()
   print("===> time to test 1 sample = " .. (time*1000) .. 'ms')

   -- printing the confusion matrix
   print(confusion)
   testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
   confusion:zero()
end

------------------------------ training ------------------ and testing -------------------------
local i_ter = opt.epoch

while i_ter > 0 do
   -- train/test
   train(trainData)
   test(testData)
   i_iter = i_iter - 1
end


-----------------------------------------------------------------------------------------------------------------------
