matio = require 'matio';
--lsp_joints = matio.load('~torch/lsp_dataset/joints.mat')

--1st lsp extended then lsp.
-- LSP 
lsp_joints = matio.load('lsp_joints.mat','joints')
lsp_joints_ex = matio.load('lspex_joints.mat','joints')
print(#lsp_joints_ex)
print(#lsp_joints)

lsp_joints_train = torch.Tensor(3,14,11000);

lsp_joints_train[{{},{},{1,10000}}] = lsp_joints_ex;
lsp_joints_train[{{},{},{10001,11000}}] = lsp_joints[{{},{},{1,1000}}];
torch.save('lsp_joints_train.t7',lsp_joints_train)
torch.save('lsp_joints_test.t7',lsp_joints[{{},{},{1001,2000}}])

lsp_joints1 = torch.load('lsp_joints_train.t7')
print(#lsp_joints1)
lsp_joints1 = torch.load('lsp_joints_test.t7')
print(#lsp_joints1)
--]]
-- FLIC
--[[

flic_joints = matio.load('FLIC_examples.mat','examples')
print(flic_joints)

--flic_joints[1][1].coords
temp = torch.Tensor(2,29,5003)
for i=1,5003 do
temp[{{},{},{i}}]=flic_joints[1][i].coords
end 
flic_joints1=temp

torch.save('flic_joints.t7',flic_joints1)

flic_joints1 = torch.load('flic_joints.t7')
print(#flic_joints1)
--]]