using System;
using System.IO;
using SimpleRBM.Common;

namespace SimpleRBM.MultiDim
{
    public class MultiDimDbnFactory : IDeepBeliefNetworkFactory<double>, IDeepBeliefNetworkFactory<float>
    {
        public IDeepBeliefNetwork<double> Create(DirectoryInfo networkDataDir, int[] appendLayers = null,
            double learningRate = 0.2,
            IExitConditionEvaluatorFactory<double> exitConditionEvaluatorFactory = null)
        {
            return new DeepBeliefNetworkD(networkDataDir, learningRate, exitConditionEvaluatorFactory, appendLayers);
        }

        public IDeepBeliefNetwork<double> Create(int[] layerSizes = null, double learningRate = 0.2,
            IExitConditionEvaluatorFactory<double> exitConditionEvaluatorFactory = null)
        {
            return new DeepBeliefNetworkD(layerSizes, learningRate, exitConditionEvaluatorFactory);
        }

        IDeepBeliefNetwork<float> IDeepBeliefNetworkFactory<float>.Create(DirectoryInfo networkDataDir,
            int[] appendLayers, float learningRate, IExitConditionEvaluatorFactory<float> exitConditionEvaluatorFactory)
        {
            return new DeepBeliefNetworkF(networkDataDir, learningRate, exitConditionEvaluatorFactory, appendLayers);
        }

        IDeepBeliefNetwork<float> IDeepBeliefNetworkFactory<float>.Create(int[] layerSizes, float learningRate,
            IExitConditionEvaluatorFactory<float> exitConditionEvaluatorFactory)
        {
            return new DeepBeliefNetworkF(layerSizes, learningRate, exitConditionEvaluatorFactory);
        }
    }
}