require 'torch'
require 'paths'
matio = require 'matio'


function loadTrainSet(maxLoad)

   fileName = "../data/train/"
   file = fileName .. "0.mat"
   f = matio.load(file)
   f.train_labels = f.train_labels:transpose(1,2)
   if maxLoad > 1 then
      for i  = 1,maxLoad-1 do
         file = fileName .. tostring(i) .. ".mat"
         temp = matio.load(file)
         f.train_patch = torch.cat(f.train_patch,temp.train_patch,1)
         f.train_labels = torch.cat(f.train_labels,temp.train_labels:transpose(1,2),1)
      end
   end

   local data = f.train_patch:type(torch.getdefaulttensortype())
   local labels = f.train_labels
   
   print("Training Data Size:")
   print(data:size())
   print("Training Labels Size:")
   print(labels:size())

   local nExample = f.train_patch:size(1)

   local dataset = {}
   dataset.data = data
   dataset.labels = labels

   function dataset:size()
      return nExample
   end

   local labelvector = torch.zeros(16)

   setmetatable(dataset, {__index = function(self, index)
			     local input = self.data[index]
			     local class = self.labels[index]
			     local label = labelvector:zero()
			     label[class[1]] = 1
			     local example = {input, label}
                                       return example
   end})

   return dataset
end

function loadTestSet(maxLoad)

   fileName = "../data/test/"
   file = fileName .. "0.mat"
   f = matio.load(file)
   f.test_labels = f.test_labels:transpose(1,2)
   if maxLoad > 1 then
      for i  = 1,maxLoad-1 do
         file = fileName .. tostring(i) .. ".mat"
         temp = matio.load(file)
         f.test_patch = torch.cat(f.test_patch,temp.test_patch,1)
         f.test_labels = torch.cat(f.test_labels,temp.test_labels:transpose(1,2),1)
      end
   end

   local data = f.test_patch:type(torch.getdefaulttensortype())
   local labels = f.test_labels

   print("Testing Data Size:")
   print(data:size())
   print("Testing Labels Size:")
   print(labels:size())

   local nExample = f.test_patch:size(1)

   local dataset = {}
   dataset.data = data
   dataset.labels = labels

   function dataset:size()
      return nExample
   end

   local labelvector = torch.zeros(16)

   setmetatable(dataset, {__index = function(self, index)
              local input = self.data[index]
              local class = self.labels[index]
              local label = labelvector:zero()
              label[class[1]] = 1
              local example = {input, label}
                                       return example
   end})

   return dataset
end