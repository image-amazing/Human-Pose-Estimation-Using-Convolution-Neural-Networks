--Original Code by Torch/tutorials
--Modified by : Vignesh for EECS442 project
local optim = require 'optim'
require 'image'

--trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
--testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))


if model then
   parameters,gradParameters = model:getParameters()
end
photo_names = torch.load('lsp_table.t7')
print(#photo_names)
mean = torch.load('mean.t7')
stdv = torch.load('stdv.t7')

testset.data = torch.Tensor(testset_size,3,h,h)
for i = 1,testset_size do
testset.data[i]=image.scale(image.load(photo_names[11000+i]),h,h)
end
for i =1,3 do
                 testset.data[{{},{i},{},{}}] = testset.data[{{},{i},{},{}}]:add(-mean[i])
                 testset.data[{{},{i},{},{}}] = testset.data[{{},{i},{},{}}]:div(stdv[i])
        end
counter = 0
min_vloss=100000
max_counter = 3
-- RMSProp 
RMSconfig = {
	learningRate = 1e-6;--0.0000005,  ----------------------------------------------------------------------------spongebob
	epsilon = 1e-8,
	weightDecay = 1e-3,
	alpha = 0.95,
	}


print('Defining RMSProp training paramters -->')

function train()
		
	
		epoch = epoch or 1
		local time = sys.clock()

		--Set model to training mode 
		--model:training() Dont need here. for models that differ in training and testing.
		local shuffle = torch.randperm(trainset_size):long() 
		
                
		--trainset.data = trainset.data[shuffle[i]]
		--trainset.labels = trainset.labels[{shuffle}]
		--print(#trainset.data)
		--labels = labels[{shuffle}]
		print('Epoch #'..epoch)
  		avgloss = 0
		--Run loop over all minibatches
		for t = 1,trainset_size,batch_size do
			print('minibatch #'..t) 
		--	iterations = (k-1)*trainset_size/batch_size + t 
			--local inputs = {}
			--local targets= {}
			--xlua.progress(t, shuffle:size(1))
 			local miniBatch = torch.zeros(batch_size,3,h,h)
			
                        local target = torch.zeros(batch_size,28)
                        for i=1,batch_size do
				miniBatch[i] = image.scale(image.load(photo_names[shuffle[t+i-1]]),h,h)--trainset.data[shuffle[t+i-1]]
				
				target[i] = trainset.labels[shuffle[t+i-1]]
			

			end
			for i =1,3 do
			miniBatch[{{},{i},{},{}}] = miniBatch[{{},{i},{},{}}]:add(-mean[i])
			miniBatch[{{},{i},{},{}}] = miniBatch[{{},{i},{},{}}]:div(stdv[i])
			end
			miniBatch = miniBatch:cuda()
			target = target:cuda()
			
	                --print(#miniBatch)
                        --print('^ mini batch ^. v target v')
			--print(#target)
			--table.insert(inputs,miniBatch)
			--table.insert(targets,target)
      			
			
			
			--parameters = nil
			--gradParameters = nil
			collectgarbage()
			 parameters,gradParameters = model:getParameters()
		        
			local feval = function(x)
				model:zeroGradParameters()
				--gradParameters:zero()
				if x~= parameters then
					parameters:copy(x)
				end
			
			
			
 			
			

				
				
					
					local output = model:forward(miniBatch)
					--print('output Shape : '.. output:size(2) .. '\ttarget shape : ' .. target:size(2))
					loss = criterion:forward(output,target)
					print('LOSS:  '..loss)
					
					--model:training()
					
					--model:zeroGradParameters()
				        df_do = criterion:backward(output,target)
					model:backward(miniBatch,df_do)
					--loss_array[(epoch-1)*batch_size+t] = loss 
					
					--parameters,gradParameters = model:getParameters()
								
   				        
				
				--Save loss,output labels
				----io.write("Loss = ",loss,"\n")
				  --if ((epoch/save_freq-math.floor(epoch/save_freq)) == 0) then
				     --torch.save('./loss/loss'..epoch..'.t7', loss)
				    -- torch.save('./model/model'..epoch..'.t7', model)
				     --model_out = output:double()
				     --torch.save('./model_out/model_out'..epoch..'.t7',model_out)
				     
				     --torch.save('./loss_array/loss_array'..epoch..'.t7',loss_array)
				  --end
				--
								
				
				return loss, gradParameters
			end
--
--
			
    			
			
			
		-- run RMSprop on mnibatch loss
			optim.rmsprop(feval,parameters,RMSconfig,{})
  			--miniBatch = nil
			--target = nil --spongebob
			--parameters = nil
			--gradParameters = nil
			local mem=collectgarbage("count")
			print('memory '..mem)


                       
		        
		end
			
                model:evaluate()
		testBatch1=testset.data:cuda()
		testBatchLabels1=testset.labels:cuda()
                testloss = criterion:forward(model:forward(testBatch1),testBatchLabels1)
		testBatch1=nil
		testBatchLabels1=nil
		collectgarbage()
                print('VALIDATION LOSS: '..testloss)
		Vloss_array[epoch]= testloss
		loss_array[epoch] = loss
                model:training()
		if ((epoch/save_freq-math.floor(epoch/save_freq)) == 0) then
			torch.save('./loss/loss'..epoch..'.t7', loss)
			torch.save('./loss_array/loss_array'..epoch..'.t7',loss_array)
			
			torch.save('./loss_array/Vloss_array'..epoch..'.t7',Vloss_array)
		end
		if(testloss < min_vloss) then
			min_vloss = testloss
			torch.save('./model/model'..epoch..'.t7', model)
			counter =0
		else 
			counter = counter +1
		end
		
			if(counter > max_counter) then
				RMSconfig.learningRate = RMSconfig.learningRate/2
				print(RMSconfig.learningRate)
				counter = 0
				torch.save('./loss/lrate.t7',RMSconfig.learningRate)
			end	
		
		time = sys.clock() - time
		print("\n==> time for epoch = " .. (time*1000) .. 'ms')
		epoch = epoch + 1

end


	

        



	
