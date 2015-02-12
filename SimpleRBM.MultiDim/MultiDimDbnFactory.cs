using System;
using System.IO;
using SimpleRBM.Common;

namespace SimpleRBM.MultiDim
{
    public class MultiDimDbnFactory : IDeepBeliefNetworkFactory<double>, IDeepBeliefNetworkFactory<float>
    {
        IDeepBeliefNetwork<double>  IDeepBeliefNetworkFactory<double>.Create(DirectoryInfo networkDataDir, ILayerDefinition[] appendLayers)
        {
            return new DeepBeliefNetworkD(networkDataDir,appendLayers);
        }

        IDeepBeliefNetwork<double>  IDeepBeliefNetworkFactory<double>.Create(ILayerDefinition[] layerSizes)
        {
            return new DeepBeliefNetworkD(layerSizes);
        }

        IDeepBeliefNetwork<float> IDeepBeliefNetworkFactory<float>.Create(DirectoryInfo networkDataDir,
            ILayerDefinition[] appendLayers)
        {
            return new DeepBeliefNetworkF(networkDataDir,  appendLayers);
        }

        IDeepBeliefNetwork<float> IDeepBeliefNetworkFactory<float>.Create(ILayerDefinition[] layerSizes)
        {
            return new DeepBeliefNetworkF(layerSizes);
        }
    }
}