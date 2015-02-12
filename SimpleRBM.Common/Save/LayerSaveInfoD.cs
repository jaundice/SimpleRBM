using System;
using System.IO;
using System.Runtime.InteropServices;

namespace SimpleRBM.Common.Save
{
    public class LayerSaveInfoD : ILayerSaveInfo<double>
    {
        public int NumVisible { get; protected set; }
        public int NumHidden { get; protected set; }
        public double[,] Weights { get; protected set; }

        public LayerSaveInfoD(int numVisible, int numHidden, double[,] weights, ActivationFunction visibleActivation, ActivationFunction hiddenActivation)
        {
            NumVisible = numVisible;
            NumHidden = numHidden;
            Weights = weights;
            VisibleActivation = visibleActivation;
            HiddenActivation = hiddenActivation;
        }

        public ActivationFunction VisibleActivation { get; protected set; }
        public ActivationFunction HiddenActivation { get; protected set; }

        public unsafe LayerSaveInfoD(string filePath)
        {
            using (var fs = File.OpenRead(filePath))
            using (var sr = new BinaryReader(fs))
            {
                NumVisible = sr.ReadInt32();
                NumHidden = sr.ReadInt32();
                VisibleActivation = (ActivationFunction)sr.ReadInt32();
                HiddenActivation = (ActivationFunction)sr.ReadInt32();
                var arr = new double[(NumVisible + 1), (NumHidden + 1)];
                var handle = GCHandle.Alloc(arr, GCHandleType.Pinned);
                var buf = (double*)handle.AddrOfPinnedObject();
                for (var i = 0; i < arr.Length; i++)
                {
                    buf[i] = sr.ReadDouble();
                }
                handle.Free();
                Weights = arr;
            }
        }

        public void Save(string filePath)
        {
            using (var fs = File.OpenWrite(filePath))
            using (var sw = new BinaryWriter(fs))
            {
                sw.Write(NumVisible);
                sw.Write(NumHidden);
                sw.Write((int)VisibleActivation);
                sw.Write((int)HiddenActivation);
                var arr = new double[Weights.Length];
                Buffer.BlockCopy(Weights, 0, arr, 0, arr.Length * sizeof(double));
                for (var i = 0; i < arr.Length; i++)
                {
                    sw.Write(arr[i]);
                }
                sw.Flush();
                sw.Close();
            }
        }
    }
}