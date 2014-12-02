using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Threading.Tasks;

namespace CudaRbm
{
    public interface IDeepBeliefNetwork<T> where T : IComparable<T>
    {
        int NumMachines { get; }
        IExitConditionEvaluatorFactory<T> ExitConditionEvaluatorFactory { get; }

        T[,] Encode(T[,] data);
        T[,] Decode(T[,] data);
        T[,] Reconstruct(T[,] data);
        T[,] DayDream(int numberOfDreams);


        T[,] Train(T[,] data, int layerPosition, out T error);
        Task AsyncTrain(T[,] data, int layerPosition);
        void TrainAll(T[,] visibleData);
        Task AsyncTrainAll(T[,] visibleData);

        IEnumerable<ILayerSaveInfo<T>> GetLayerSaveInfos();

        event EventHandler<EpochEventArgs<T>> EpochEnd;
        event EventHandler<EpochEventArgs<T>> TrainEnd;
    }

    public interface ILayerSaveInfo<T>
    {
        int NumVisible { get; }
        int NumHidden { get; }
        T[,] Weights { get; }
        void Save(string filePath);
    }

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

    public class EpochEventArgs<T> : EventArgs
    {
        public int Epoch { get; set; }
        public T Error { get; set; }
    }
}