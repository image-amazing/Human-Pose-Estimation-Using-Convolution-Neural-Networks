
require 'nn'
-- can replace nn by cudann in next 2 lines
local SpatialConvolution = nn.SpatialConvolution--lib[1]
local SpatialMaxPooling = nn.SpatialMaxPooling--lib[2]


 function create_net()

 model = nn.Sequential()
model:add(SpatialConvolution(3,96,11,11,4,4,2,2))       --224 -> 55  original pad 2. change to 5
model:add(nn.ReLU(true))

model:add(SpatialMaxPooling(3,3,2,2))                   -- 55 ->  27
model:add(SpatialConvolution(96,256,5,5,1,1,2,2))       --  27 -> 27
model:add(nn.ReLU(true))
model:add(SpatialMaxPooling(3,3,2,2))                   --  27 ->  13
model:add(SpatialConvolution(256,384,3,3,1,1,1,1))      --  13 ->  13
model:add(nn.ReLU(true))
model:add(SpatialConvolution(384,384,3,3,1,1,1,1))      --  13 ->  13
model:add(nn.ReLU(true))
model:add(SpatialConvolution(384,256,3,3,1,1,1,1))      --  13 ->  13
model:add(nn.ReLU(true))
model:add(SpatialMaxPooling(3,3,2,2))                   -- 13 -> 6


model:add(nn.View(256*6*6))
model:add(nn.ReLU(true))
--model:add(nn.Dropout(0.5))
model:add(nn.Linear(256*6*6, 4096))
model:add(nn.ReLU(true))

--model:add(nn.Dropout(0.5))
model:add(nn.Linear(4096, 4096))
model:add(nn.ReLU(true))
--model:add(nn.Threshold(0, 1e-6)) --?
model:add(nn.Linear(4096, 28)) -- to change
--model:add(nn.MSECriterion()) --to change: put loss outside model





return {
  model = model,
  regime = {
    epoch        = {1,    19,   30,   44,   53  },
    learningRate = {1e-2, 5e-3, 1e-3, 5e-4, 1e-4},
    weightDecay  = {5e-4, 5e-4, 0,    0,    0   }
  }
}
end
