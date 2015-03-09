using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Windows;
using Cudafy.Host;
using Cudafy.Maths.RAND;
using CudaNN.DeepBelief.LayerBuilders;

namespace CudaNN.DeepBelief.ViewModels
{
    public class LayerBuilderViewModel : DependencyObject
    {
        public static readonly DependencyProperty LayerConstructionInfoProperty =
          DependencyProperty.Register("LayerConstructionInfo", typeof(ObservableCollection<ConstructLayerBase>),
              typeof(LayerBuilderViewModel), new PropertyMetadata(default(ObservableCollection<ConstructLayerBase>)));


        public LayerBuilderViewModel()
        {
            LayerConstructionInfo = new ObservableCollection<ConstructLayerBase>();
        }

        public ObservableCollection<ConstructLayerBase> LayerConstructionInfo
        {
            get { return Dispatcher.InvokeIfRequired(() => (ObservableCollection<ConstructLayerBase>)GetValue(LayerConstructionInfoProperty)).Result; }
            set { Dispatcher.InvokeIfRequired(() => SetValue(LayerConstructionInfoProperty, value)).Wait(); }

        }

        public IEnumerable<IAdvancedRbmCuda<double>> CreateLayers(GPGPU gpu, GPGPURAND rand)
        {
            return LayerConstructionInfo.Select((createLayerBase, i) =>
            {
                ConstructNewLayer newLayer = createLayerBase as ConstructNewLayer;
                if (newLayer != null)
                {
                    newLayer.LayerIndex = i;
                }
                return CreateLayers(createLayerBase, gpu, rand);
            });
        }

        private IAdvancedRbmCuda<double> CreateLayers(ConstructLayerBase constructLayerBase, GPGPU gpu, GPGPURAND rand)
        {
            var loadLayerInfo = constructLayerBase as LoadLayerInfo;
            if (loadLayerInfo != null)
            {
                return Load(loadLayerInfo, gpu, rand);
            }

            var conBin = constructLayerBase as ConstructBinaryLayer;
            if (conBin != null)
            {
                return ConstructBinaryLayer(conBin, gpu, rand);
            }

            var conLin = constructLayerBase as ConstructLinearHiddenLayer;
            if (conLin != null)
            {
                return ConstructLinearLayer(conLin, gpu, rand);
            }

            throw new NotImplementedException();
        }

        private IAdvancedRbmCuda<double> ConstructLinearLayer(ConstructLinearHiddenLayer conLin, GPGPU gpu,
            GPGPURAND rand)
        {
            return new CudaAdvancedRbmLinearHidden(gpu, rand, conLin.LayerIndex, conLin.NumVisibleNeurons,
                conLin.NumHiddenNeurons, conLin.WeightCost, conLin.InitialMomentum, conLin.FinalMomentum,
                conLin.WeightInitializationStDev, conLin.TrainRandStDev);
        }

        private IAdvancedRbmCuda<double> ConstructBinaryLayer(ConstructBinaryLayer conBin, GPGPU gpu, GPGPURAND rand)
        {
            return new CudaAdvancedRbmBinary(gpu, rand, conBin.LayerIndex, conBin.NumVisibleNeurons,
                conBin.NumHiddenNeurons, conBin.ConvertActivationsToStates, conBin.WeightCost, conBin.InitialMomentum,
                conBin.FinalMomentum, conBin.EncodingNoiseLevel, conBin.DecodingNoiseLevel,
                conBin.WeightInitializationStDev);
        }


        private IAdvancedRbmCuda<double> Load(LoadLayerInfo info, GPGPU gpu, GPGPURAND rand)
        {
            return CudaAdvancedRbmBase.Deserialize(info.Path, gpu, rand);
        }
    }
}