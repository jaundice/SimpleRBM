using System;
using System.Collections.ObjectModel;
using System.Configuration;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Windows;
using CudaNN.DeepBelief.DataIO;
using SimpleRBM.Demo;
using SimpleRBM.Demo.Util;
using Size = System.Windows.Size;
#if USEFLOAT
using TElement = System.Single;
#else
using TElement = System.Double;
#endif
namespace CudaNN.DeepBelief.ViewModels
{
    public class ImageDataConfigViewModel : DataConfigViewModelBase
    {
        public static readonly DependencyProperty ImageSizeProperty =
            DependencyProperty.Register("ImageSize", typeof(Size),
                typeof(ImageDataConfigViewModel), new PropertyMetadata(default(Size)));

        public static readonly DependencyProperty LabelsProperty =
            DependencyProperty.Register("Labels", typeof(ObservableCollection<string>),
                typeof(ImageDataConfigViewModel), new PropertyMetadata(default(ObservableCollection<string>)));


        public static readonly DependencyProperty UseGrayCodeForLabelsProperty =
            DependencyProperty.Register("UseGrayCodeForLabels", typeof(bool),
                typeof(ImageDataConfigViewModel), new PropertyMetadata(default(bool)));

        public DataTransformationTypes[] AllAvailableTransformationTypes
        {
            get { return new[] { DataTransformationTypes.DivideBy255, DataTransformationTypes.Subtract128Divide127 }; }
        }

        public Size ImageSize
        {
            get { return (Size)GetValue(ImageSizeProperty); }
            set { SetValue(ImageSizeProperty, value); }
        }

        public ObservableCollection<string> Labels
        {
            get { return (ObservableCollection<string>)GetValue(LabelsProperty); }
            set { SetValue(LabelsProperty, value); }
        }

        public bool UseGrayCodeForLabels
        {
            get { return (bool)GetValue(UseGrayCodeForLabelsProperty); }
            set { SetValue(UseGrayCodeForLabelsProperty, value); }
        }

        public override DataContainerType ContainerType
        {
            get { return DataContainerType.Directory; }
        }

        public override string FileExtensionFilter
        {
            get { return "*.jpg"; }
        }

        public override void OnTrainingDataChanged(string path)
        {
            if (!string.IsNullOrEmpty(path) && Directory.Exists(path))
            {
                var di = new DirectoryInfo(path);
                TotalTrainingRecordsAvailableCount =
                    di.EnumerateFiles(FileExtensionFilter, SearchOption.AllDirectories).Count();

                Labels = new ObservableCollection<string>(di.EnumerateDirectories().Select(a => a.Name));

                FileInfo fs = di.EnumerateFiles(FileExtensionFilter, SearchOption.AllDirectories).FirstOrDefault();
                if (fs != null)
                {
                    using (Image bmp = Image.FromFile(fs.FullName))
                    {
                        ImageSize = new Size(bmp.Width, bmp.Height);
                        DataWidth = (int)ImageSize.Width * (int)ImageSize.Height;
                        LabelWidth = UseGrayCodeForLabels
                            ? new FieldGrayEncoder<string>(Labels).ElementsRequired
                            : Labels.Count;
                    }
                }
            }
        }

        public override void OnTestDataChanged(string path)
        {
            if (!string.IsNullOrEmpty(path))
            {
                var di = new DirectoryInfo(path);
                TotalTestRecordsAvailableCount =
                    di.EnumerateFiles(FileExtensionFilter, SearchOption.AllDirectories).Count();
            }
        }


        public override void GetDataReaders(out DataReaderBase<TElement> trainingReader,
            out DataReaderBase<TElement> validationReader, out DataReaderBase<TElement> testReader)
        {
            ImageUtils.ConvertPixel<TElement> imgConverter;
            Func<TElement, TElement> sourceToTarget;
            Func<TElement, TElement> targetToSource;

           
#if USEFLOAT
                        imgConverter = ImageUtils.ConvertRGBToGreyIntF;
#else
            imgConverter = ImageUtils.ConvertRGBToGreyIntD;
#endif

            switch (DataTransformationType)
            {
                case DataTransformationTypes.NoTransform:
                case DataTransformationTypes.DivideBy255:
                    {
                        sourceToTarget = a => a / (TElement)255;
                        targetToSource = a => a * (TElement)255;
                        break;
                    }
                case DataTransformationTypes.Subtract128Divide127:
                    {
                        sourceToTarget = a => (a - (TElement)128) / (TElement)127;
                        targetToSource = a => (a * (TElement)128) + (TElement)127;
                        break;
                    }
                default: throw new NotImplementedException();
            }
            var extensions = new[] { ".jpg" };

            if (RandomizeTrainingData)
            {
                trainingReader = new RandomImageReader<TElement>(TrainingDataPath, UseGrayCodeForLabels, DataWidth, Labels,
                    extensions, TotalTrainingRecordsAvailableCount, sourceToTarget, targetToSource, imgConverter);
            }
            else
            {
                trainingReader = new SequentialImageReader<TElement>(TrainingDataPath, UseGrayCodeForLabels, DataWidth,
                    Labels,
                    extensions, SkipTrainingRecordCount, TotalTrainingRecordsAvailableCount, sourceToTarget, targetToSource, imgConverter);
            }
            if (RandomizeValidationData)
            {
                validationReader = new RandomImageReader<TElement>(TrainingDataPath, UseGrayCodeForLabels, DataWidth,
                    Labels,
                    extensions, TotalTrainingRecordsAvailableCount, sourceToTarget, targetToSource, imgConverter);
            }
            else
            {
                validationReader = new SequentialImageReader<TElement>(TrainingDataPath, UseGrayCodeForLabels, DataWidth,
                    Labels, extensions, SkipValidationRecordCount, TotalTrainingRecordsAvailableCount, sourceToTarget, targetToSource, imgConverter);
            }
            if (RandomizeTestData)
            {
                testReader = new RandomImageReader<TElement>(TestDataPath, UseGrayCodeForLabels, DataWidth, Labels,
                    extensions, TotalTestRecordsAvailableCount, sourceToTarget, targetToSource, imgConverter);
            }
            else
            {
                testReader = new SequentialImageReader<TElement>(TestDataPath, UseGrayCodeForLabels, DataWidth, Labels,
                    extensions, SkipTestRecordCount, TotalTestRecordsAvailableCount, sourceToTarget, targetToSource, imgConverter);
            }

        }

        protected override void OnPropertyChanged(DependencyPropertyChangedEventArgs e)
        {
            base.OnPropertyChanged(e);
            if (e.Property == UseGrayCodeForLabelsProperty)
            {
                OnTrainingDataChanged(TrainingDataPath);
            }
        }
    }
}