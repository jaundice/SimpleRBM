using System;
using System.IO;

namespace SimpleRBM.Common
{
    public interface IDeepBeliefNetworkFactory<T> where T : struct, IComparable<T>
    {
        IDeepBeliefNetwork<T> Create(DirectoryInfo networkDataDir, ILayerDefinition[] appendLayers);


        IDeepBeliefNetwork<T> Create(ILayerDefinition[] layerSizes);
    }
}