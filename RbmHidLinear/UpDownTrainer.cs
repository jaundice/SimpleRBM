/*
http://www.cs.toronto.edu/~hinton/absps/ncfast.pdf
\% UP-DOWN ALGORITHM
\%
\% the data and all biases are row vectors.
\% the generative model is: lab <--> top <--> pen --> hid --> vis
\% the number of units in layer foo is numfoo
\% weight matrices have names fromlayer tolayer
\% "rec" is for recognition biases and "gen" is for generative
\% biases.
\% for simplicity, the same learning rate, r, is used everywhere.
\% PERFORM A BOTTOM-UP PASS TO GET WAKE/POSITIVE PHASE
\% PROBABILITIES AND SAMPLE STATES
wakehidprobs = logistic(data*vishid + hidrecbiases);
wakehidstates = wakehidprobs > rand(1, numhid);
wakepenprobs = logistic(wakehidstates*hidpen + penrecbiases);
wakepenstates = wakepenprobs > rand(1, numpen);
wakeopprobs = logistic(wakepenstates*pentop + targets*labtop +
topbiases);
wakeopstates = wakeopprobs > rand(1, numtop);
\% POSITIVE PHASE STATISTICS FOR CONTRASTIVE DIVERGENCE
poslabtopstatistics = targets’ * waketopstates;
pospentopstatistics = wakepenstates’ * waketopstates;
\% PERFORM numCDiters GIBBS SAMPLING ITERATIONS USING THE TOP LEVEL
\% UNDIRECTED ASSOCIATIVE MEMORY
negtopstates = waketopstates; \% to initialize loop
for iter=1:numCDiters
negpenprobs = logistic(negtopstates*pentop’ + pengenbiases);
negpenstates = negpenprobs > rand(1, numpen);
neglabprobs = softmax(negtopstates*labtop’ + labgenbiases);
negtopprobs = logistic(negpenstates*pentop+neglabprobs*labtop+
topbiases);
negtopstates = negtopprobs > rand(1, numtop));
end;
\% NEGATIVE PHASE STATISTICS FOR CONTRASTIVE DIVERGENCE
negpentopstatistics = negpenstates’*negtopstates;
neglabtopstatistics = neglabprobs’*negtopstates;
\% STARTING FROM THE END OF THE GIBBS SAMPLING RUN, PERFORM A
\% TOP-DOWN GENERATIVE PASS TO GET SLEEP/NEGATIVE PHASE
\% PROBABILITIES AND SAMPLE STATES
sleeppenstates = negpenstates;
sleephidprobs = logistic(sleeppenstates*penhid + hidgenbiases);
sleephidstates = sleephidprobs > rand(1, numhid);
sleepvisprobs = logistic(sleephidstates*hidvis + visgenbiases);
\% PREDICTIONS
psleeppenstates = logistic(sleephidstates*hidpen + penrecbiases);
psleephidstates = logistic(sleepvisprobs*vishid + hidrecbiases);
pvisprobs = logistic(wakehidstates*hidvis + visgenbiases);
phidprobs = logistic(wakepenstates*penhid + hidgenbiases);
\% UPDATES TO GENERATIVE PARAMETERS
hidvis = hidvis + r*poshidstates’*(data-pvisprobs);
visgenbiases = visgenbiases + r*(data - pvisprobs);
penhid = penhid + r*wakepenstates’*(wakehidstates-phidprobs);
hidgenbiases = hidgenbiases + r*(wakehidstates - phidprobs);
\% UPDATES TO TOP LEVEL ASSOCIATIVE MEMORY PARAMETERS
labtop = labtop + r*(poslabtopstatistics-neglabtopstatistics);
labgenbiases = labgenbiases + r*(targets - neglabprobs);
pentop = pentop + r*(pospentopstatistics - negpentopstatistics);
pengenbiases = pengenbiases + r*(wakepenstates - negpenstates);
topbiases = topbiases + r*(waketopstates - negtopstates);
\%UPDATES TO RECOGNITION/INFERENCE APPROXIMATION PARAMETERS
hidpen = hidpen + r*(sleephidstates’*(sleeppenstatespsleeppenstates));
penrecbiases = penrecbiases + r*(sleeppenstates-psleeppenstates);
vishid = vishid + r*(sleepvisprobs’*(sleephidstatespsleephidstates));
hidrecbiases = hidrecbiases + r*(sleephidstates-psleephidstates);
 */


using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Text;
using System.Threading.Tasks;
using Cudafy.Host;
using ICSharpCode.Decompiler.Ast.Transforms;
using SimpleRBM.Cuda;
#if USEFLOAT
using TElement = System.Single;
using xxx = SimpleRBM.Cuda.CudaRbmF;

#else
using TElement = System.Double;
using xxx = SimpleRBM.Cuda.CudaRbmD;

#endif
namespace CudaNN
{
    public class UpDownTrainer
    {
        public UpDownTrainer(ICudaNetwork<TElement> net)
        {
            Network = net;
        }

        public ICudaNetwork<double> Network { get; protected set; }

        public void UpDownTrain(TElement[,] visData, TElement[,] labels, TElement weightLearningRate,
            TElement hidBiasLearningRate, TElement visBiasLearningRate, int cdIterations)
        {
            var gpu = Network.Machines[0].GPU;
            var rand = Network.Machines[0].GPURAND;
            using (var data = gpu.Upload(visData))
            using (var targets = gpu.Upload(labels))
            {
                using (var wakehidstates = Network.Machines[0].Encode(data))
                using (var wakepenstates = Network.Machines[1].Encode(wakehidstates))
                using (var combined = Combine(gpu, wakepenstates, targets))
                using (var waketopstates = Network.Machines[2].Encode(combined))
                using (var targetsTrans = targets.Transpose())
                using (var poslabtopstatistics = targetsTrans.Multiply(waketopstates))
                using (var wakepenstatesTrans = wakepenstates.Transpose())
                using (var pospentopstatistics = wakepenstatesTrans.Multiply(waketopstates))
                {
                    var negtopstates = waketopstates;
                    for (var i = 0; i < cdIterations; i++)
                    {
                       //todo:wip
                    }
                }
            }
        }

        private Matrix2D<TElement> Combine(GPGPU gpu, Matrix2D<TElement> wakepenstates, Matrix2D<TElement> targets)
        {
            var ret = gpu.AllocateNoSet<TElement>(wakepenstates.GetLength(0),
                wakepenstates.GetLength(1) + targets.GetLength(1));

            ret.InsertValuesFrom(0, 0, wakepenstates);
            ret.InsertValuesFrom(0, wakepenstates.GetLength(1), targets);
            return ret;
        }
    }
}
