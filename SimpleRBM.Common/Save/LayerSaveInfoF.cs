using System;
using System.IO;
using System.Runtime.InteropServices;

namespace SimpleRBM.Common.Save
{
    public class LayerSaveInfoF : ILayerSaveInfo<float>
    {
        public int NumVisible { get; protected set; }
        public int NumHidden { get; protected set; }
        public float[,] Weights { get; protected set; }

        public LayerSaveInfoF(int numVisible, int numHidden, float[,] weights)
        {
            NumVisible = numVisible;
            NumHidden = numHidden;
            Weights = weights;
        }

        public unsafe LayerSaveInfoF(string filePath)
        {
            using (var fs = File.OpenRead(filePath))
            using (var sr = new BinaryReader(fs))
            {
                NumVisible = sr.ReadInt32();
                NumHidden = sr.ReadInt32();
                var arr = new float[(NumVisible + 1), (NumHidden + 1)];
                var handle = GCHandle.Alloc(arr, GCHandleType.Pinned);
                var buf = (float*)handle.AddrOfPinnedObject();
                for (var i = 0; i < arr.Length; i++)
                {
                    buf[i] = sr.ReadSingle();
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
                var arr = new float[Weights.Length];
                Buffer.BlockCopy(Weights, 0, arr, 0, arr.Length * sizeof(float));
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