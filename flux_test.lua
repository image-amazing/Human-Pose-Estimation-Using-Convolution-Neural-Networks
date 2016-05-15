-- Tests the final model, on the test set. 

require('torch')
require('nn')
require('cunn')
require('cutorch')
math = require('math')
--matio = require('matio')
nlabels = 14
model = torch.load('./analysis/test3/model/model191500.t7')
criterion = nn.MSECriterion():cuda();
--


--SANITY CHECK
--input = torch.rand(3,224,224):cuda();
--output = model:forward(input);
--print('output size')
--print(#output)
--gradInput = model:backward(input, torch.CudaTensor(28))
--criterion:forward(output, torch.CudaTensor(28))
--

--CREATE testset

labels = torch.load('lsp_joints_test.t7'); -- 
labels1 = labels[{{1,2},{},{}}]:clone() -- 2x14xn 

--Scale labels in accordance to image scaling

original_image_size = torch.load('lsp_images_size_test.t7'); 
testset_size = original_image_size:size(1)
for i=1,testset_size do 
	--print(original_image_size[i])
	labels1[{{1},{},{i}}]:div(original_image_size[{{i},{2}}]:squeeze() /224) --width
	labels1[{{2},{},{i}}]:div(original_image_size[{{i},{1}}]:squeeze() /224) --height
end


labels2 = labels1:reshape(28,testset_size) --28xn -- (x1,x2,x3...x14,y1,y2,y3...y14)'
labels3 = labels2:transpose(1,2) --nx28    					-- spongebob
labels = labels3[{{1,1000},{1,28}}]							-- spongebob
print(#labels)
data1 = torch.load('lsp_images_test.t7') --10x3x224x224 			 -- spongebob
data = data1[{{1,1000},{},{},{}}]						 -- spongebob
testset_size = data:size(1)
testset_size = 1000                						 -- spongebob
testset = {data = data:cuda(), labels = labels:cuda()}				
--testset.data:double() -- convert the data from a ByteTensor to a DoubleTensor.
--testset.labels:double()

--PREPROCESS DATA
mean = {} -- store the mean, to normalize the test set in the future
stdv  = {} -- store the standard-deviation for the future
for i=1,3 do -- over each image channel
    mean[i] = testset.data[{ {}, {i}, {}, {}  }]:mean() -- mean estimation
    print('Channel ' .. i .. ', Mean: ' .. mean[i])
    testset.data[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction
    
    stdv[i] = testset.data[{ {}, {i}, {}, {}  }]:std() -- std estimation
    print('Channel ' .. i .. ', Standard Deviation: ' .. stdv[i])
    testset.data[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
end


model_out = model:forward(testset.data)
loss = criterion:forward(model_out, testset.labels)
print(loss)
print(model_out[{{1},{}}])

torch.save('./test_model_out/test_label.t7',model_out)


for i=1,testset_size do 
	--print(original_image_size[i])
	model_out[{{i},{1,14}}]:div(224 /original_image_size[{{i},{2}}]:squeeze()) --width
	model_out[{{i},{15,28}}]:div(224 /original_image_size[{{i},{1}}]:squeeze()) --height
end


----- PDJ Percentage of Detected Joints Evaluation Metric
det = torch.zeros(nlabels)
not_det = torch.zeros(nlabels)
torso = torch.zeros(testset_size)
length = 0
thresh = 1
--det = 0
--not_det = 0
for i=1,testset_size do
    --det = 0
    --not_det = 0
    torso[i] = torch.sqrt((labels[i][4] - labels[i][9])^2 + (labels[i][nlabels+4] - labels[i][nlabels+9])^2)  -- Find torso joints Left hip = 4, Right hip = 3, Left Shoulder = 10, Right Shoulder = 9   
    for j=1,nlabels do
	length = torch.sqrt((labels[i][j]-model_out[i][j])^2 + (labels[i][j+nlabels]-model_out[i][j+nlabels])^2)  
        if length<thresh*torso[i] then
		det[j] = det[j] + 1
	else
		not_det[j] = not_det[j] + 1
	end
    end
end
pdj = det*100/testset_size
notpdj = not_det*100/testset_size
print('PDJ = ',pdj)
--pdj = det/(testset_size*2*nlabels)
--notpdj = not_det/(testset_size*2*nlabels)
 

----- PCP Percentage of Correct Parts Evaluation Metric
dist = torch.zeros(14)
length = torch.zeros(nlabels-1)
pcpdet = torch.zeros(9)
pcp = torch.zeros(9)
for i=1,testset_size do
	for j=1,nlabels-1 do
      		length[j] = torch.sqrt((labels[i][j]-labels[i][j+1])^2 + (labels[i][nlabels+j]-labels[i][nlabels+j+1])^2)
	end
	
	dist[1] = torch.sqrt((labels[i][1]-model_out[i][1])^2 + (labels[i][1+nlabels]-model_out[i][1+nlabels])^2)
        dist[2] = torch.sqrt((labels[i][2]-model_out[i][2])^2 + (labels[i][2+nlabels]-model_out[i][2+nlabels])^2)
        dist[3] = torch.sqrt((labels[i][3]-model_out[i][3])^2 + (labels[i][3+nlabels]-model_out[i][3+nlabels])^2)
        dist[4] = torch.sqrt((labels[i][4]-model_out[i][4])^2 + (labels[i][4+nlabels]-model_out[i][4+nlabels])^2)
        dist[5] = torch.sqrt((labels[i][5]-model_out[i][5])^2 + (labels[i][5+nlabels]-model_out[i][5+nlabels])^2)
        dist[6] = torch.sqrt((labels[i][6]-model_out[i][6])^2 + (labels[i][6+nlabels]-model_out[i][6+nlabels])^2)
        dist[7] = torch.sqrt((labels[i][7]-model_out[i][7])^2 + (labels[i][7+nlabels]-model_out[i][7+nlabels])^2)
        dist[8] = torch.sqrt((labels[i][8]-model_out[i][8])^2 + (labels[i][8+nlabels]-model_out[i][8+nlabels])^2)
        dist[9] = torch.sqrt((labels[i][9]-model_out[i][9])^2 + (labels[i][9+nlabels]-model_out[i][9+nlabels])^2)
        dist[10] = torch.sqrt((labels[i][10]-model_out[i][10])^2 + (labels[i][10+nlabels]-model_out[i][10+nlabels])^2)
        dist[11] = torch.sqrt((labels[i][11]-model_out[i][11])^2 + (labels[i][11+nlabels]-model_out[i][11+nlabels])^2)
        dist[12] = torch.sqrt((labels[i][12]-model_out[i][12])^2 + (labels[i][12+nlabels]-model_out[i][12+nlabels])^2)
        dist[13] = torch.sqrt((labels[i][13]-model_out[i][13])^2 + (labels[i][13+nlabels]-model_out[i][13+nlabels])^2)
        dist[14] = torch.sqrt((labels[i][14]-model_out[i][14])^2 + (labels[i][14+nlabels]-model_out[i][14+nlabels])^2)




        if (dist[1]+dist[2])<length[1] then   --- rankle-rknee
                pcpdet[1]=pcpdet[1]+1
        end

        if (dist[2]+dist[3])<length[2] then   --- rknee-rhip
                pcpdet[2]=pcpdet[2]+1
        end

        if (dist[4]+dist[5])<length[4] then   --- lhip-lknee
                pcpdet[3]=pcpdet[3]+1
        end

        if (dist[5]+dist[6])<length[5] then   --- lknee-lankle
                pcpdet[4]=pcpdet[4]+1
        end

       if (dist[7]+dist[8])<length[7] then   --- rwrist-relbow
                pcpdet[5]=pcpdet[5]+1
        end

       if (dist[8]+dist[9])<length[8] then   --- relbow-rshoulder
                pcpdet[6]=pcpdet[6]+1
        end

       if (dist[10]+dist[11])<length[10] then   --- lshoulder-lelbow
                pcpdet[7]=pcpdet[7]+1
        end

       if (dist[11]+dist[12])<length[11] then   --- lelbow-lwrist
                pcpdet[8]=pcpdet[8]+1
        end

       if (dist[13]+dist[14])<length[13] then   --- neck-head
                pcpdet[9]=pcpdet[9]+1
        end
		
end

pcp = pcpdet*100/testset_size
print('PCP = ',pcp) 
model_out = model_out:double()

torch.save('./test_model_out/test_loss.t7',loss)
torch.save('./test_model_out/test_label.t7',model_out)
torch.save('./test_model_out/pcp.t7',pcp)
torch.save('./test_model_out/pdj10.t7',pdj)
