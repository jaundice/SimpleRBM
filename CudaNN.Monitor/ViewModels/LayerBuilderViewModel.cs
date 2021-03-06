﻿using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Windows;
using Cudafy.Host;
using Cudafy.Maths.RAND;
using CudaNN.DeepBelief.LayerBuilders;
#if USEFLOAT
using TElement = System.Single;
#else
using TElement = System.Double;
#endif
namespace CudaNN.DeepBelief.ViewModels
{
    public class LayerBuilderViewModel : DependencyObject
    {
        public static readonly DependencyProperty LayerConstructionInfoProperty =
            DependencyProperty.Register("LayerConstructionInfo", typeof(ObservableCollection<ConstructLayerBase>),
                typeof(LayerBuilderViewModel), new PropertyMetadata(default(ObservableCollection<ConstructLayerBase>)));


        public static readonly DependencyProperty StartTrainLayerProperty =
            DependencyProperty.Register(" StartTrainLayer", typeof(ConstructLayerBase),
                typeof(LayerBuilderViewModel), new PropertyMetadata(default(ConstructLayerBase)));


        public static readonly DependencyProperty StartTrainLayerIndexProperty =
            DependencyProperty.Register(" StartTrainLayerIndex", typeof(int),
                typeof(LayerBuilderViewModel), new PropertyMetadata(default(int)));

        public LayerBuilderViewModel()
        {
            LayerConstructionInfo = new ObservableCollection<ConstructLayerBase>();
        }

        public ObservableCollection<ConstructLayerBase> LayerConstructionInfo
        {
            get
            {
                return
                    Dispatcher.InvokeIfRequired(
                        () => (ObservableCollection<ConstructLayerBase>)GetValue(LayerConstructionInfoProperty)).Result;
            }
            set { Dispatcher.InvokeIfRequired(() => SetValue(LayerConstructionInfoProperty, value)).Wait(); }
        }

        public ConstructLayerBase StartTrainLayer
        {
            get { return (ConstructLayerBase)GetValue(StartTrainLayerProperty); }
            set { SetValue(StartTrainLayerProperty, value); }
        }

        public int StartTrainLayerIndex
        {
            get { return (int)GetValue(StartTrainLayerIndexProperty); }
            set { SetValue(StartTrainLayerIndexProperty, value); }
        }

        public IEnumerable<IAdvancedRbmCuda<TElement>> CreateLayers(GPGPU gpu, GPGPURAND rand)
        {
            return LayerConstructionInfo.Select((createLayerBase, i) =>
            {
                var newLayer = createLayerBase as ConstructNewLayer;
                if (newLayer != null)
                {
                    newLayer.LayerIndex = i;
                }
                return CreateLayers(createLayerBase, gpu, rand);
            });
        }

        private IAdvancedRbmCuda<TElement> CreateLayers(ConstructLayerBase constructLayerBase, GPGPU gpu, GPGPURAND rand)
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

        private IAdvancedRbmCuda<TElement> ConstructLinearLayer(ConstructLinearHiddenLayer conLin, GPGPU gpu,
            GPGPURAND rand)
        {
            return new CudaAdvancedRbmLinearHidden(gpu, rand, conLin.LayerIndex, conLin.NumVisibleNeurons,
                conLin.NumHiddenNeurons, conLin.WeightCost, conLin.InitialMomentum, conLin.FinalMomentum,
                conLin.WeightInitializationStDev, conLin.TrainRandStDev, conLin.MomentumIncrementStep);
        }

        private IAdvancedRbmCuda<TElement> ConstructBinaryLayer(ConstructBinaryLayer conBin, GPGPU gpu, GPGPURAND rand)
        {
            return new CudaAdvancedRbmBinary(gpu, rand, conBin.LayerIndex, conBin.NumVisibleNeurons,
                conBin.NumHiddenNeurons, conBin.ConvertActivationsToStates, conBin.WeightCost, conBin.InitialMomentum,
                conBin.FinalMomentum, conBin.EncodingNoiseLevel, conBin.DecodingNoiseLevel,
                conBin.WeightInitializationStDev, conBin.MomentumIncrementStep);
        }


        private IAdvancedRbmCuda<TElement> Load(LoadLayerInfo info, GPGPU gpu, GPGPURAND rand)
        {
            return CudaAdvancedRbmBase.Deserialize(info.Path, gpu, rand);
        }

        protected override void OnPropertyChanged(DependencyPropertyChangedEventArgs e)
        {
            base.OnPropertyChanged(e);
            if (e.Property == StartTrainLayerProperty)
            {
                StartTrainLayerIndex = LayerConstructionInfo.IndexOf((ConstructLayerBase)e.NewValue);
            }
        }
    }
}