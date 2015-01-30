using System;
using System.IO;
using SimpleRBM.Common;

namespace SimpleRBM.MultiDim
{
    public class MultiDimDbnFactory : IDeepBeliefNetworkFactory<double>, IDeepBeliefNetworkFactory<float>
    {
        public IDeepBeliefNetwork<double> Create(DirectoryInfo networkDataDir, ILayerDefinition[] appendLayers,
            ILearningRateCalculator<double> learningRateCalculator ,
            IExitConditionEvaluatorFactory<double> exitConditionEvaluatorFactory = null)
        {
            return new DeepBeliefNetworkD(networkDataDir, learningRateCalculator, exitConditionEvaluatorFactory, appendLayers);
        }

        public IDeepBeliefNetwork<double> Create(ILayerDefinition[] layerSizes, ILearningRateCalculator<double> learningRateCalculator,
            IExitConditionEvaluatorFactory<double> exitConditionEvaluatorFactory = null)
        {
            return new DeepBeliefNetworkD(layerSizes, learningRateCalculator, exitConditionEvaluatorFactory);
        }

        IDeepBeliefNetwork<float> IDeepBeliefNetworkFactory<float>.Create(DirectoryInfo networkDataDir,
            ILayerDefinition[] appendLayers, ILearningRateCalculator<float> learningRateCalculator, IExitConditionEvaluatorFactory<float> exitConditionEvaluatorFactory)
        {
            return new DeepBeliefNetworkF(networkDataDir, learningRateCalculator, exitConditionEvaluatorFactory, appendLayers);
        }

        IDeepBeliefNetwork<float> IDeepBeliefNetworkFactory<float>.Create(ILayerDefinition[] layerSizes, ILearningRateCalculator<float> learningRate,
            IExitConditionEvaluatorFactory<float> exitConditionEvaluatorFactory)
        {
            return new DeepBeliefNetworkF(layerSizes, learningRate, exitConditionEvaluatorFactory);
        }
    }
}