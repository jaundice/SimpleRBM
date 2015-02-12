//https://github.com/sinshu/wav_utility/blob/master/WavUtil.cs
using System;
using System.IO;
using System.Threading.Tasks;

namespace SimpleRBM.Demo.WavUtil
{
    public static class WavUtil
    {
        private const int RiffHeader = 0x46464952;
        private const int WaveHeader = 0x45564157;
        private const int FmtHeader = 0x20746D66;
        private const int DataHeader = 0x61746164;

        public static double[][] Read(string fileName, out int sampFreq)
        {
            using (var fs = new FileStream(fileName, FileMode.Open, FileAccess.Read))
            {
                using (var br = new BinaryReader(fs))
                {
                    if (br.ReadInt32() != RiffHeader)
                    {
                        throw new Exception("Invalid Riff Header");
                    }
                    br.ReadInt32();
                    if (br.ReadInt32() != WaveHeader)
                    {
                        throw new Exception("Invalid Wav Header");
                    }

                    while (br.ReadInt32() != FmtHeader)
                    {
                        br.BaseStream.Seek(br.ReadInt32() + 1 >> 1 << 1, SeekOrigin.Current);
                    }

                    int numChannels;
                    {
                        var chunk = new byte[br.ReadInt32()];
                        br.Read(chunk, 0, chunk.Length);
                        ReadFmtChunk(chunk, out numChannels, out sampFreq);
                        if ((chunk.Length & 1) > 0) br.ReadByte();
                    }

                    while (br.ReadInt32() != DataHeader)
                    {
                        br.BaseStream.Seek(br.ReadInt32() + 1 >> 1 << 1, SeekOrigin.Current);
                    }

                    {
                        var chunk = new byte[br.ReadInt32()];
                        br.Read(chunk, 0, chunk.Length);
                        return ReadDataChunk(chunk, numChannels);
                    }
                }
            }
        }

        public static double[][] Read(string fileName)
        {
            int sampFreq;
            return Read(fileName, out sampFreq);
        }

        public static double[] ReadMono(string fileName, out int sampFreq)
        {
            return Read(fileName, out sampFreq)[0];
        }

        public static double[] ReadMono(string fileName)
        {
            int sampFreq;
            return ReadMono(fileName, out sampFreq);
        }

        public static void Write(string fileName, double[][] data, int sampFreq)
        {
            int numChannels = data.Length;
            int pcmLength = data[0].Length;
            int dataChunkSize = 2 * numChannels * pcmLength;
            int riffChunkSize = dataChunkSize + 36;
            using (var fs = new FileStream(fileName, FileMode.Create, FileAccess.Write))
            {
                using (var bw = new BinaryWriter(fs))
                {
                    bw.Write(RiffHeader);
                    bw.Write(riffChunkSize);
                    bw.Write(WaveHeader);
                    bw.Write(FmtHeader);
                    bw.Write(16);
                    bw.Write((short)1);
                    bw.Write((short)numChannels);
                    bw.Write(sampFreq);
                    bw.Write(2 * numChannels * sampFreq);
                    bw.Write((short)(2 * numChannels));
                    bw.Write((short)16);
                    bw.Write(DataHeader);
                    bw.Write(dataChunkSize);
                    bool clipped = false;
                    for (int t = 0; t < pcmLength; t++)
                    {
                        for (int ch = 0; ch < numChannels; ch++)
                        {
                            var s = (int)Math.Floor(32768 * data[ch][t]);
                            if (s < short.MinValue)
                            {
                                s = short.MinValue;
                                clipped = true;
                            }
                            else if (s > short.MaxValue)
                            {
                                s = short.MaxValue;
                                clipped = true;
                            }
                            bw.Write((short)s);
                        }
                    }
                    if (clipped)
                    {
                        Console.Error.WriteLine("Wav is Clipped");
                    }
                }
            }
        }

        public static void Write(string fileName, double[] data, int sampFreq)
        {
            Write(fileName, new[] { data }, sampFreq);
        }

        public static void Normalize(params double[][] data)
        {
            double max = double.MinValue;
            foreach (var d in data)
            {
                foreach (double s in d)
                {
                    if (max < Math.Abs(s))
                    {
                        max = Math.Abs(s);
                    }
                }
            }
            for (int ch = 0; ch < data.Length; ch++)
            {
                for (int t = 0; t < data[ch].Length; t++)
                {
                    data[ch][t] = 0.99 * data[ch][t] / max;
                }
            }
        }

        private static void ReadFmtChunk(byte[] chunk, out int numChannels, out int sampFreq)
        {
            using (var ms = new MemoryStream(chunk))
            {
                using (var br = new BinaryReader(ms))
                {
                    int formatId = br.ReadInt16();
                    if (formatId != 1)
                    {
                        throw new Exception("Invalid Fmt Chunk");
                    }
                    numChannels = br.ReadInt16();
                    sampFreq = br.ReadInt32();
                    br.ReadInt32();
                    br.ReadInt16();
                    int quantBit = br.ReadInt16();
                    if (quantBit != 16)
                    {
                        throw new Exception("Invalid quantBit");
                    }
                }
            }
            // Console.WriteLine("numChannels: " + numChannels);
            // Console.WriteLine("sampFreq: " + sampFreq);
        }

        private static double[][] ReadDataChunk(byte[] chunk, int numChannels)
        {
            int pcmLength = chunk.Length / (2 * numChannels);
            var data = new double[numChannels][];
            for (int ch = 0; ch < numChannels; ch++)
            {
                data[ch] = new double[pcmLength];
            }
            using (var ms = new MemoryStream(chunk))
            {
                using (var br = new BinaryReader(ms))
                {
                    for (int t = 0; t < pcmLength; t++)
                    {
                        for (int ch = 0; ch < numChannels; ch++)
                        {
                            data[ch][t] = br.ReadInt16() / 32768.0;
                        }
                    }
                }
            }
            return data;
        }
    }

    public static class WavUtil<T>
    {
        public static void SaveWavData(T[,] data, int sourceRow, string path, Func<T, double> converter)
        {
            WavUtil.Write(path, ConvertData(data, sourceRow, converter), 44100);
        }

        private static double[][] ConvertData(T[,] data, int sourceRow, Func<T, double> converter)
        {
            var arr = new double[2][];
            Parallel.For(0, arr.Length, a => arr[a] = new double[data.GetLength(1) / 2]);

            Parallel.For(0, data.GetLength(1) / 2, a =>
            {
                arr[0][a] = converter(data[sourceRow, 2 * a]);
                arr[1][a] = converter(data[sourceRow, (2 * a) + 1]);
            });

            return arr;
        }
    }
}