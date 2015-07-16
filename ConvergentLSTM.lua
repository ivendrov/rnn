--[[ ConvergentLSTM ]]--
-- Like GenLSTM, but with the output gate only dependent on the input
-- and making sure that a zero input vector = zero output

local ConvergentLSTM, parent = torch.class('nn.ConvergentLSTM', 'nn.GenLSTM')

function ConvergentLSTM:buildOutputGate()
   -- Note : gate expects an input table : {input, output(t-1), cell(t-1)}
   local gate = nn.Sequential()
   gate:add(nn.SelectTable(1))
   gate:add(nn.LinearNoBias(self.inputSize, self.outputSize))
   gate:add(nn.Tanh())
   gate:add(nn.Abs())
   return gate
end