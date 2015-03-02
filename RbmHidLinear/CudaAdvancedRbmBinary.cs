using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.Serialization;
using Cudafy.Host;
using Cudafy.Maths.RAND;
using SimpleRBM.Common;
using SimpleRBM.Cuda;
/*
http://www.cs.toronto.edu/~hinton/code/rbm.m
% Version 1.000 
%
% Code provided by Geoff Hinton and Ruslan Salakhutdinov 
%
% Permission is granted for anyone to copy, use, modify, or distribute this
% program and accompanying programs and documents for any purpose, provided
% this copyright notice is retained and prominently displayed, along with
% a note saying that the original programs are available from our
% web page.
% The programs and documents are distributed without any warranty, express or
% implied.  As the programs were written for research purposes only, they have
% not been tested to the degree that would be advisable in any important
% application.  All use of these programs is entirely at the user's own risk.

% This program trains Restricted Boltzmann Machine in which
% visible, binary, stochastic pixels are connected to
% hidden, binary, stochastic feature detectors using symmetrically
% weighted connections. Learning is done with 1-step Contrastive Divergence.   
% The program assumes that the following variables are set externally:
% maxepoch  -- maximum number of epochs
% numhid    -- number of hidden units 
% batchdata -- the data that is divided into batches (numcases numdims numbatches)
% restart   -- set to 1 if learning starts from beginning 

epsilonw      = 0.1;   % Learning rate for weights 
epsilonvb     = 0.1;   % Learning rate for biases of visible units 
epsilonhb     = 0.1;   % Learning rate for biases of hidden units 
weightcost  = 0.0002;   
initialmomentum  = 0.5;
finalmomentum    = 0.9;

[numcases numdims numbatches]=size(batchdata);

if restart ==1,
  restart=0;
  epoch=1;

% Initializing symmetric weights and biases. 
  vishid     = 0.1*randn(numdims, numhid);
  hidbiases  = zeros(1,numhid);
  visbiases  = zeros(1,numdims);

  poshidprobs = zeros(numcases,numhid);
  neghidprobs = zeros(numcases,numhid);
  posprods    = zeros(numdims,numhid);
  negprods    = zeros(numdims,numhid);
  vishidinc  = zeros(numdims,numhid);
  hidbiasinc = zeros(1,numhid);
  visbiasinc = zeros(1,numdims);
  batchposhidprobs=zeros(numcases,numhid,numbatches);
end

for epoch = epoch:maxepoch,
 fprintf(1,'epoch %d\r',epoch); 
 errsum=0;
 for batch = 1:numbatches,
 fprintf(1,'epoch %d batch %d\r',epoch,batch); 

%%%%%%%%% START POSITIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  data = batchdata(:,:,batch);
  poshidprobs = 1./(1 + exp(-data*vishid - repmat(hidbiases,numcases,1)));    
  batchposhidprobs(:,:,batch)=poshidprobs;
  posprods    = data' * poshidprobs;
  poshidact   = sum(poshidprobs);
  posvisact = sum(data);

%%%%%%%%% END OF POSITIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  poshidstates = poshidprobs > rand(numcases,numhid);

%%%%%%%%% START NEGATIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  negdata = 1./(1 + exp(-poshidstates*vishid' - repmat(visbiases,numcases,1)));
  neghidprobs = 1./(1 + exp(-negdata*vishid - repmat(hidbiases,numcases,1)));    
  negprods  = negdata'*neghidprobs;
  neghidact = sum(neghidprobs);
  negvisact = sum(negdata); 

%%%%%%%%% END OF NEGATIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  err= sum(sum( (data-negdata).^2 ));
  errsum = err + errsum;

   if epoch>5,
     momentum=finalmomentum;
   else
     momentum=initialmomentum;
   end;

%%%%%%%%% UPDATE WEIGHTS AND BIASES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    vishidinc = momentum*vishidinc + ...
                epsilonw*( (posprods-negprods)/numcases - weightcost*vishid);
    visbiasinc = momentum*visbiasinc + (epsilonvb/numcases)*(posvisact-negvisact);
    hidbiasinc = momentum*hidbiasinc + (epsilonhb/numcases)*(poshidact-neghidact);

    vishid = vishid + vishidinc;
    visbiases = visbiases + visbiasinc;
    hidbiases = hidbiases + hidbiasinc;

%%%%%%%%%%%%%%%% END OF UPDATES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

  end
  fprintf(1, 'epoch %4i error %6.1f  \n', epoch, errsum); 
end;
*/
#if USEFLOAT
using TElement = System.Single;
using xxx = SimpleRBM.Cuda.CudaRbmF;

#else
using TElement = System.Double;
using xxx = SimpleRBM.Cuda.CudaRbmD;

#endif

namespace CudaNN
{
    [Serializable]
    public class CudaAdvancedRbmBinary : CudaAdvancedRbmBase
    {
        public bool ConvertActivationsToStates { get; protected set; }

        private TElement _decodingNoiseLevel;
        private TElement _encodingNoiseLevel;

        public CudaAdvancedRbmBinary(GPGPU gpu, GPGPURAND rand, int layerIndex, int numVisibleNeurons,
            int numHiddenNeurons, bool convertActivationsToStates,
            TElement weightcost = (TElement) 0.0002,
            TElement initialMomentum = (TElement) 0.5, TElement finalMomentum = (TElement) 0.9,
            TElement encodingNoiseLevel = (TElement) 1, TElement decodingNoiseLevel = (TElement) 1, TElement weightInitializationStDev = (TElement)0.01)
            : base(
                gpu, rand, layerIndex, numVisibleNeurons, numHiddenNeurons, /*epsilonw, epsilonvb, epsilonhb,*/
                weightcost,
                initialMomentum, finalMomentum, weightInitializationStDev)
        {
            ConvertActivationsToStates = convertActivationsToStates;
            _decodingNoiseLevel = decodingNoiseLevel;
            _encodingNoiseLevel = encodingNoiseLevel;
        }


        public override Matrix2D<TElement> Encode(Matrix2D<TElement> data)
        {
            var state = State;
            Wake();
            int numcases = data.GetLength(0);
            using (Matrix2D<TElement> tiledHiddenBiases = AsCuda.HiddenBiases.RepMatRows(numcases))
            {
                using (Matrix2D<TElement> datavishid = data.Multiply(AsCuda.Weights))
                using (Matrix2D<TElement> poshidprobs = datavishid.Subtract(tiledHiddenBiases))
                {
                    poshidprobs.LogisticInPlace();
                    using (
                        Matrix2D<TElement> rand = AsCuda.GPU.UniformDistribution(AsCuda.GPURAND, numcases,
                            NumHiddenNeurons, (TElement) _encodingNoiseLevel))
                    {
                        SetState(state);
                        //end positive phase
                        return poshidprobs.GreaterThan(rand);
                    }
                }
            }
        }

        public override Matrix2D<TElement> Decode(Matrix2D<TElement> activations)
        {
            var state = State;
            Wake();

            int numcases = activations.GetLength(0);
            using (Matrix2D<TElement> tiledVisibleBiases = AsCuda.VisibleBiases.RepMatRows(numcases))
            using (Matrix2D<TElement> weightsTransposed = AsCuda.Weights.Transpose())
            using (
                Matrix2D<TElement> poshidstatesweightstransposed =
                    activations.Multiply(weightsTransposed))
            {
                Matrix2D<TElement> negdata = poshidstatesweightstransposed.Subtract(tiledVisibleBiases);
                negdata.LogisticInPlace();

                if (ConvertActivationsToStates)
                {
                    using (negdata)
                    using (
                        var rnd = AsCuda.GPU.UniformDistribution(AsCuda.GPURAND, numcases, NumVisibleNeurons,
                            (TElement) _decodingNoiseLevel))
                    {
                        negdata = negdata.GreaterThan(rnd);
                    }
                }

                SetState(state);
                return negdata;
            }
        }

        //public override void GreedyBatchedTrain(Matrix2D<TElement> data, int batchSize,
        //    IExitConditionEvaluator<TElement> exitConditionEvaluator,
        //    ILearningRateCalculator<TElement> weightLearningRateCalculator,
        //    ILearningRateCalculator<TElement> hidBiasLearningRateCalculator,
        //    ILearningRateCalculator<TElement> visBiasLearningRateCalculator)
        //{
        //    var state = State;
        //    Wake();
        //    int numcases = data.GetLength(0);

        //    exitConditionEvaluator.Start();
        //    var datasets = PartitionDataAsMatrices(data, batchSize);
        //    try
        //    {
        //        Stopwatch sw = new Stopwatch();
        //        int epoch;
        //        TElement error;
        //        for (epoch = 0;; epoch++)
        //        {
        //            sw.Restart();
        //            error =
        //                datasets.Sum(block => BatchedTrainEpoch(block.Item1, block.Item2, block.Item3, epoch, numcases,
        //                    weightLearningRateCalculator, hidBiasLearningRateCalculator, visBiasLearningRateCalculator));

        //            OnEpochComplete(new EpochEventArgs<TElement>()
        //            {
        //                Epoch = epoch,
        //                Error = error,
        //                Layer = LayerIndex
        //            });

        //            if (exitConditionEvaluator.Exit(epoch, error, sw.Elapsed))
        //                break;
        //        }

        //        OnTrainComplete(new EpochEventArgs<TElement>()
        //        {
        //            Epoch = epoch,
        //            Error = error,
        //            Layer = LayerIndex
        //        });
        //    }
        //    finally
        //    {
        //        foreach (var dataset in datasets)
        //        {
        //            dataset.Item1.Dispose();
        //            dataset.Item2.Dispose();
        //            dataset.Item3.Dispose();
        //        }
        //    }

        //    exitConditionEvaluator.Stop();
        //    SetState(state);
        //}

        //public override void GreedyTrain(Matrix2D<TElement> data,
        //    IExitConditionEvaluator<TElement> exitConditionEvaluator,
        //    ILearningRateCalculator<TElement> weightLearningRateCalculator,
        //    ILearningRateCalculator<TElement> hidBiasLearningRateCalculator,
        //    ILearningRateCalculator<TElement> visBiasLearningRateCalculator)
        //{
        //    var state = State;
        //    Wake();
        //    int numcases = data.GetLength(0);
        //    exitConditionEvaluator.Start();
        //    var sw = new Stopwatch();
        //    using (Matrix2D<TElement> dataTransposed = data.Transpose())
        //    using (Matrix2D<TElement> posvisact = data.SumColumns())
        //    {
        //        int epoch;
        //        TElement error;
        //        for (epoch = 0;; epoch++)
        //        {
        //            sw.Restart();
        //            error = BatchedTrainEpoch(data, dataTransposed, posvisact, epoch, numcases,
        //                weightLearningRateCalculator, hidBiasLearningRateCalculator, visBiasLearningRateCalculator);

        //            OnEpochComplete(new EpochEventArgs<TElement>()
        //            {
        //                Epoch = epoch,
        //                Error = error,
        //                Layer = LayerIndex
        //            });

        //            if (exitConditionEvaluator.Exit(epoch, error, sw.Elapsed))
        //                break;
        //        }

        //        OnTrainComplete(new EpochEventArgs<TElement>()
        //        {
        //            Epoch = epoch,
        //            Error = error,
        //            Layer = LayerIndex
        //        });
        //    }
        //    SetState(state);
        //}

        protected override TElement BatchedTrainEpoch(Matrix2D<TElement> data, Matrix2D<TElement> dataTransposed,
            Matrix2D<TElement> posvisact, int epoch, int numcases,
            TElement weightLearningRate,
            TElement hidBiasLearningRate,
            TElement visBiasLearningRate)
        {
            TElement error;

            var batchCases = data.GetLength(0);


            //start positive phase
            using (Matrix2D<TElement> tiledHiddenBiases = AsCuda.HiddenBiases.RepMatRows(batchCases))
            {
                Matrix2D<TElement> poshidstates, poshidact, posprods;
                using (Matrix2D<TElement> datavishid = data.Multiply(AsCuda.Weights))
                using (Matrix2D<TElement> poshidprobs = datavishid.Subtract(tiledHiddenBiases))
                {
                    poshidprobs.LogisticInPlace();
                    poshidact = poshidprobs.SumColumns();
                    posprods = dataTransposed.Multiply(poshidprobs);
                    using (
                        Matrix2D<TElement> rand = AsCuda.GPU.UniformDistribution(AsCuda.GPURAND, batchCases,
                            NumHiddenNeurons, (TElement) _encodingNoiseLevel))
                    {
                        //end positive phase
                        poshidstates = poshidprobs.GreaterThan(rand);
                    }
                }

                //start negative phase
                Matrix2D<TElement> negdata, negprods, neghidact, negvisact;
                using (poshidstates)
                using (Matrix2D<TElement> tiledVisibleBiases = AsCuda.VisibleBiases.RepMatRows(batchCases))
                using (Matrix2D<TElement> weightsTransposed = AsCuda.Weights.Transpose())
                using (
                    Matrix2D<TElement> poshidstatesweightstransposed =
                        poshidstates.Multiply(weightsTransposed))
                {
                    negdata = poshidstatesweightstransposed.Subtract(tiledVisibleBiases);
                    negdata.LogisticInPlace();

                    using (Matrix2D<TElement> negdataWeights = negdata.Multiply(AsCuda.Weights))
                    using (Matrix2D<TElement> neghiddenprobs = negdataWeights.Subtract(tiledHiddenBiases))
                    {
                        neghiddenprobs.LogisticInPlace();
                        using (var negdataTransposed = negdata.Transpose())
                        {
                            negprods = negdataTransposed.Multiply(neghiddenprobs);
                            neghidact = neghiddenprobs.SumColumns();
                            negvisact = negdata.SumColumns();
                        }
                    }
                }

                //end negative phase
                using (negdata)
                using (Matrix2D<TElement> delta = data.Subtract(negdata))
                {
                    delta.PowInPlace((TElement) 2);
                    using (var errCols = delta.SumColumns())
                    using (var errrows = errCols.SumRows())
                        error = errrows.CopyLocal()[0, 0];
                }

                TElement momentum = epoch > 5 ? FinalMomentum : InitialMomentum;

                using (negprods)
                using (posprods)
                using (Matrix2D<TElement> momentumvishidinc = WeightInc.Multiply(momentum))
                using (Matrix2D<TElement> posprodsminusnegprods = posprods.Subtract(negprods))
                using (Matrix2D<TElement> weightcostWeight = AsCuda.Weights.Multiply(WeightCost))
                {
                    posprodsminusnegprods.MultiplyInPlace((TElement) 1/(TElement) numcases);
                    posprodsminusnegprods.SubtractInPlace(weightcostWeight);
                    posprodsminusnegprods.MultiplyInPlace(weightLearningRate);
                    WeightInc.Dispose();
                    _vishidinc = momentumvishidinc.Add(posprodsminusnegprods);
                }

                using (negvisact)
                using (Matrix2D<TElement> momentumvisbiasinc = VisibleBiasInc.Multiply(momentum))
                using (Matrix2D<TElement> posvisactminusnegvisact = posvisact.Subtract(negvisact))
                {
                    posvisactminusnegvisact.MultiplyInPlace(
                        visBiasLearningRate/numcases);
                    VisibleBiasInc.Dispose();
                    _visbiasinc = momentumvisbiasinc.Add(posvisactminusnegvisact);
                }

                using (neghidact)
                using(poshidact)
                using (Matrix2D<TElement> momentumhidbiasinc = HiddenBiasInc.Multiply(momentum))
                using (Matrix2D<TElement> poshidactminusneghidact = poshidact.Subtract(neghidact))
                {
                    poshidactminusneghidact.MultiplyInPlace(
                        hidBiasLearningRate/numcases);
                    HiddenBiasInc.Dispose();
                    _hidbiasinc = momentumhidbiasinc.Add(poshidactminusneghidact);
                }

                AsCuda.Weights.AddInPlace(WeightInc);
                AsCuda.VisibleBiases.AddInPlace(VisibleBiasInc);
                AsCuda.HiddenBiases.AddInPlace(HiddenBiasInc);
            }
            return error;
        }

        //public override void GreedyBatchedTrainMem(Matrix2D<TElement> data, int batchSize,
        //    IExitConditionEvaluator<TElement> exitConditionEvaluator,
        //    ILearningRateCalculator<TElement> weightLearningRateCalculator,
        //    ILearningRateCalculator<TElement> hidBiasLearningRateCalculator,
        //    ILearningRateCalculator<TElement> visBiasLearningRateCalculator)
        //{
        //    var state = State;


        //    exitConditionEvaluator.Start();
        //    int numcases = data.GetLength(0);

        //    List<Tuple<TElement[,], TElement[,], TElement[,]>> datasets;

        //    Suspend(); //free memory for processing dataset
        //    using (data)
        //    {
        //        datasets = PartitionDataAsArrays(data, batchSize);
        //    }
        //    Wake();

        //    Stopwatch sw = new Stopwatch();
        //    int epoch;
        //    TElement error;
        //    for (epoch = 0;; epoch++)
        //    {
        //        sw.Restart();
        //        error = datasets.Sum(block =>
        //        {
        //            using (var d = AsCuda.GPU.Upload(block.Item1))
        //            using (var t = AsCuda.GPU.Upload(block.Item2))
        //            using (var p = AsCuda.GPU.Upload(block.Item3))
        //                return BatchedTrainEpoch(d, t, p, epoch, numcases,
        //                    weightLearningRateCalculator, hidBiasLearningRateCalculator,
        //                    visBiasLearningRateCalculator);
        //        });

        //        OnEpochComplete(new EpochEventArgs<TElement>()
        //        {
        //            Epoch = epoch,
        //            Error = error,
        //            Layer = LayerIndex
        //        });

        //        if (exitConditionEvaluator.Exit(epoch, error, sw.Elapsed))
        //            break;
        //    }

        //    OnTrainComplete(new EpochEventArgs<TElement>()
        //    {
        //        Epoch = epoch,
        //        Error = error,
        //        Layer = LayerIndex
        //    });
        //    exitConditionEvaluator.Stop();
        //    SetState(state);
        //}


        protected override void SaveSpecific(SerializationInfo info)
        {
            info.AddValue("convertActivationsToStates", ConvertActivationsToStates);
            info.AddValue("decNoise", _decodingNoiseLevel);
            info.AddValue("encNoise", _encodingNoiseLevel);
        }

        protected override void LoadSpecific(SerializationInfo info)
        {
            base.LoadSpecific(info);
            ConvertActivationsToStates = info.GetBoolean("convertActivationsToStates");
            _encodingNoiseLevel = (TElement) info.GetValue("encNoise", typeof (TElement));
            _decodingNoiseLevel = (TElement) info.GetValue("decNoise", typeof (TElement));
        }

       
    }
}