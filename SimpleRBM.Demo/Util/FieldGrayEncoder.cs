using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Security.Cryptography;
using System.Threading.Tasks;

namespace SimpleRBM.Demo.Util
{
    //public static class FieldGrayEncoder
    //{
    //    public static T[,] EncodeLabels<TLabel, T>(TLabel[] labels, int numDistinctLabelOptions = 0)
    //    {
    //        List<TLabel> distinct = labels.Distinct().OrderBy(a => a).ToList();
    //        Dictionary<TLabel, int> indices = distinct.ToDictionary(a => a, distinct.IndexOf);

    //        var on = (T)Convert.ChangeType(1.0, typeof(T));

    //        if (numDistinctLabelOptions == 0)
    //            numDistinctLabelOptions = distinct.Count;

    //        var array = new T[labels.Length, numDistinctLabelOptions];
    //        for (int i = 0; i < labels.Length; i++)
    //        {
    //            array[i, indices[labels[i]]] = on;
    //        }
    //        return array;
    //    }

    //    public static T[,] EncodeLabels<TLabel, T>(TLabel[] labels, TLabel[] distinctLabelOptions)
    //    {
    //        List<TLabel> distinct = distinctLabelOptions.OrderBy(a => a).ToList();
    //        Dictionary<TLabel, int> indices = distinct.ToDictionary(a => a, distinct.IndexOf);

    //        var on = (T)Convert.ChangeType(1.0, typeof(T));


    //        var array = new T[labels.Length, distinctLabelOptions.Length];
    //        for (int i = 0; i < labels.Length; i++)
    //        {
    //            array[i, indices[labels[i]]] = on;
    //        }
    //        return array;
    //    }
    //}


    public class FieldGrayEncoder<TLabel>
    {
        private readonly Dictionary<TLabel, ulong> _labelToIndexMap;
        private Dictionary<ulong, TLabel> _indexToLabelMap;
        private readonly ulong _numElements;

        public int ElementsRequired
        {
            get
            {
                if (_numElements < byte.MaxValue)
                    return sizeof(byte) * 8;
                if (_numElements < ushort.MaxValue)
                    return sizeof(ushort) * 8;
                if (_numElements < uint.MaxValue)
                    return sizeof(uint) * 8;
                if (_numElements < ulong.MaxValue)
                    return sizeof(ulong) * 8;

                throw new ArgumentOutOfRangeException();
            }
        }
        public FieldGrayEncoder(IEnumerable<TLabel> allPossibleLabels)
        {
            var distinct = allPossibleLabels.Distinct().OrderBy(a => a).ToList();
            _numElements = (ulong)distinct.LongCount();
            var temp = distinct.Select((a, i) =>

                new
                {
                    el = a,
                    ind = (ulong)i

                }).ToList();
            _labelToIndexMap = temp.ToDictionary(a => a.el, b => b.ind);
            _indexToLabelMap = temp.ToDictionary(a => a.ind, b => b.el);
        }

        public TLabel Decode<T>(T[,] target, int targetRow, int rowOffset, T onValue, T offValue)
        {
            var width = ElementsRequired;

            if (width == 8)
            {
                return DecodeU8<T>(target, targetRow, rowOffset, onValue, offValue);
            }
            if (width == 16)
            {
                return DecodeU16<T>(target, targetRow, rowOffset, onValue, offValue);
            }
            if (width == 32)
            {
                return DecodeU32<T>(target, targetRow, rowOffset, onValue, offValue);
            }
            if (width == 64)
            {
                return DecodeU64<T>(target, targetRow, rowOffset, onValue, offValue);
            }
            throw new NotImplementedException();
        }

        private TLabel DecodeU8<T>(T[,] target, int targetRow, int rowOffset, T onValue, T offValue)
        {
            TLabel lbl;

            return _indexToLabelMap.TryGetValue(GrayCodeU8.ReadBits(target, targetRow, rowOffset, onValue, offValue).Code, out lbl) ? lbl : default (TLabel);
        }
        private TLabel DecodeU16<T>(T[,] target, int targetRow, int rowOffset, T onValue, T offValue)
        {
            TLabel lbl;

            return _indexToLabelMap.TryGetValue(GrayCodeU16.ReadBits(target, targetRow, rowOffset, onValue, offValue).Code, out lbl) ? lbl : default(TLabel);
        }
        private TLabel DecodeU32<T>(T[,] target, int targetRow, int rowOffset, T onValue, T offValue)
        {
            throw new NotImplementedException();
        }
        private TLabel DecodeU64<T>(T[,] target, int targetRow, int rowOffset, T onValue, T offValue)
        {
            throw new NotImplementedException();
        }

        public T[,] Encode<T>(TLabel[] labels, T onValue, T offValue)
        {
            var width = ElementsRequired;
            T[,] res = new T[labels.GetLength(0), width];

            if (width == 8)
            {
                EncodeU8<T>(labels, res, 0, onValue, offValue);
            }
            if (width == 16)
            {
                EncodeU16<T>(labels, res, 0, onValue, offValue);
            }
            if (width == 32)
            {
                EncodeU32<T>(labels, res, 0, onValue, offValue);
            }
            if (width == 64)
            {
                EncodeU64<T>(labels, res, 0, onValue, offValue);
            }

            return res;
        }

        public void Encode<T>(TLabel label, T[,] target, int targetRow, int startColumn, T on, T off)
        {
            var width = ElementsRequired;

            if (width == 8)
            {
                EncodeU8<T>(label, target, targetRow, startColumn, on, off);
            }
            if (width == 16)
            {
                EncodeU16<T>(label, target, targetRow, startColumn, on, off);
            }
            if (width == 32)
            {
                EncodeU32<T>(label, target, targetRow, startColumn, on, off);
            }
            if (width == 64)
            {
                EncodeU64<T>(label, target, targetRow, startColumn, on, off);
            }

        }

        private void EncodeU64<T>(TLabel label, T[,] target, int targetRow, int startColumn, T on, T off)
        {
            var idx = (ulong)_labelToIndexMap[label];
            var code = (GrayCodeU64)idx;
            var setBits = GrayCodeU64.GetSetBits(code, on, off);
            for (var j = 0; j < setBits.Length; j++)
            {
                target[targetRow, j + startColumn] = setBits[j];
            }
        }

        private void EncodeU32<T>(TLabel label, T[,] target, int targetRow, int startColumn, T on, T off)
        {
            var idx = (uint)_labelToIndexMap[label];
            var code = (GrayCodeU32)idx;
            GrayCodeU32.SetBits(code, target, targetRow, startColumn, on, off);
        }

        private void EncodeU16<T>(TLabel label, T[,] target, int targetRow, int startColumn, T on, T off)
        {
            var idx = (ushort)_labelToIndexMap[label];
            var code = (GrayCodeU16)idx;
            GrayCodeU16.SetBits(code, target, targetRow, startColumn, on, off);

        }

        private void EncodeU8<T>(TLabel label, T[,] target, int targetRow, int startColumn, T on, T off)
        {
            var idx = (byte)_labelToIndexMap[label];
            var code = (GrayCodeU8)idx;
            GrayCodeU8.SetBits(code, target, targetRow, startColumn, on, off);
        }


        private void EncodeU8<T>(TLabel[] labels, T[,] target, int startRowOffset, T on, T off)
        {
            Parallel.For(0, labels.GetLength(0), i => EncodeU8<T>(labels[i], target, i, startRowOffset, @on, off));
        }

        private void EncodeU16<T>(TLabel[] labels, T[,] target, int targetRowOffset, T on, T off)
        {
            Parallel.For(0, labels.GetLength(0), i => EncodeU16<T>(labels[i], target, i, targetRowOffset, @on, off));
        }

        private void EncodeU32<T>(TLabel[] labels, T[,] target, int startRowOffset, T on, T off)
        {
            Parallel.For(0, labels.GetLength(0), i => EncodeU32<T>(labels[i], target, i, startRowOffset, @on, off));
        }

        private void EncodeU64<T>(TLabel[] labels, T[,] target, int startRowOffset, T on, T off)
        {
            Parallel.For(0, labels.GetLength(0), i => EncodeU64<T>(labels[i], target, i, startRowOffset, @on, off));
        }
    }
}