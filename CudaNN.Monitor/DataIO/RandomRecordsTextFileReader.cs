using System;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;

namespace CudaNN.DeepBelief.DataIO
{
    public class RandomRecordsTextFileReader<T> : TextFileReaderBase<T> where T : IComparable<T>, IEquatable<T>
    {
        public RandomRecordsTextFileReader(LineReader<T> labelReader, LineReader<T> dataReader, bool firstLineIsHeaders,
            int totalRecordCount, string filePath, char fieldSeparator)
            : base(labelReader, dataReader, firstLineIsHeaders, totalRecordCount, filePath, fieldSeparator)
        {
        }

        public override T[,] ReadWithLabels(int count, out T[,] labelsEncoded, out string[] labels)
        {
            using (FileStream fs = File.OpenRead(FilePath))
            using (var sr = new StreamReader(fs))
            {
                if (FirstLineIsHeaders)
                    sr.ReadLine();

                var data = new T[count, DataWidth];
                labelsEncoded = new T[count, LabelDataWidth];
                var lab = new string[count];

                double cutOff = ((double)count) / TotalRecordCount;
                var rnd = new Random(DateTime.Now.Millisecond);
                int lineNo = 0;
                string line;

                for (line = sr.ReadLine();
                    !sr.EndOfStream && lineNo < count && !string.IsNullOrWhiteSpace(line);
                    line = sr.ReadLine())
                {
                    if (rnd.NextDouble() <= cutOff)
                    {
                        string[] lineParts = line.Split(FieldSeparator);
                        LabelReader.CopyToTarget(labelsEncoded, lineParts, lineNo);
                        lab[lineNo] = string.Join(",", LabelReader.ReadTargetLineContents(labelsEncoded, lineNo));
                        DataReader.CopyToTarget(data, lineParts, lineNo);
                        lineNo++;
                    }
                }


                if (lineNo < count + 1)
                {
                    data = SubMatrix(data, 0, 0, lineNo - 1, DataWidth);
                    labelsEncoded = SubMatrix(labelsEncoded, 0, 0, lineNo - 1, LabelDataWidth);
                    var lbl = SubVector(lab, 0, lineNo - 1);
                    lab = lbl;
                }
                labels = lab;
                return data;
            }
        }

        private static T[] SubVector<T>(T[] lab, int startIndex, int count)
        {
            var lbl = new T[count];
            Parallel.For(0, count, i => lbl[i] = lab[i + startIndex]);
            return lbl;
        }

        public override T[,] Read(int count)
        {
            using (FileStream fs = File.OpenRead(FilePath))
            using (var sr = new StreamReader(fs))
            {
                if (FirstLineIsHeaders)
                    sr.ReadLine();

                var data = new T[count, DataWidth];

                double cutOff = ((double)count) / TotalRecordCount;
                var rnd = new Random(DateTime.Now.Millisecond);
                int lineNo = 0;
                string line;

                for (line = sr.ReadLine();
                    !sr.EndOfStream && lineNo < count && !string.IsNullOrWhiteSpace(line);
                    line = sr.ReadLine())
                {
                    if (rnd.NextDouble() <= cutOff)
                    {
                        string[] lineParts = line.Split(FieldSeparator);
                        DataReader.CopyToTarget(data, lineParts, lineNo);
                        lineNo++;
                    }
                }

                //todo:if we end up with fewer records than requested copy into a smaller array;

                if (lineNo < count + 1)
                {
                    data = SubMatrix(data, 0, 0, lineNo - 1, DataWidth);
                }

                return data;
            }
        }

        private T[,] SubMatrix(T[,] data, int startRow, int startCol, int numRows, int numCols)
        {
            var ret = new T[numRows, numCols];
            Parallel.For(0, numRows, i => Parallel.For(0, numCols, j =>
            {
                ret[i, j] = data[i + startRow, j + startCol];
            }));
            return ret;
        }


        public override IList<T[,]> ReadWithLabels(int count, int batchSize, out IList<T[,]> labelsEncoded,
            out IList<string[]> labels)
        {
            List<string[]> lbls = new List<string[]>();
            List<T[,]> coded = new List<T[,]>();
            List<T[,]> data = new List<T[,]>();


            double cutOff = ((double)count) / TotalRecordCount;
            var rnd = new Random(DateTime.Now.Millisecond);

            using (FileStream fs = File.OpenRead(FilePath))
            using (var sr = new StreamReader(fs))
            {
                if (FirstLineIsHeaders)
                    sr.ReadLine();

                var dat = new T[batchSize, DataWidth];
                var cod = new T[batchSize, LabelDataWidth];
                var lb = new string[batchSize];

                data.Add(dat);
                coded.Add(cod);
                lbls.Add(lb);
                int newbatchSize = 0;
                int lineNo = 0;
                string line;
                int j = 0;
                for (line = sr.ReadLine(), j = 0;
                    !sr.EndOfStream && lineNo < count && !string.IsNullOrWhiteSpace(line);
                    line = sr.ReadLine())
                {
                    if (rnd.NextDouble() <= cutOff)
                    {
                        string[] lineParts = line.Split(FieldSeparator);
                        LabelReader.CopyToTarget(cod, lineParts, j);
                        lb[j] = string.Join(",", LabelReader.ReadTargetLineContents(cod, j));
                        DataReader.CopyToTarget(dat, lineParts, j);

                        lineNo++;
                        j++;

                        if (j == dat.GetLength(0))
                        {
                            j = 0;
                            newbatchSize = lineNo + batchSize < count ? batchSize : count - lineNo;
                            if (newbatchSize > 0)
                            {
                                dat = new T[newbatchSize, DataWidth];
                                lb = new string[newbatchSize];
                                cod = new T[newbatchSize, LabelDataWidth];
                                data.Add(dat);
                                lbls.Add(lb);
                                coded.Add(cod);
                            }
                        }
                    }
                }

                if (j < newbatchSize + 1)
                {
                    var lastIndex = data.Count - 1;
                    if (j - 1 < 1)
                    {
                        data.RemoveAt(lastIndex);
                        lbls.RemoveAt(lastIndex);
                    }
                    else
                    {
                        data[lastIndex] = SubMatrix(data[lastIndex], 0, 0, j - 1, DataWidth);
                        coded[lastIndex] = SubMatrix(coded[lastIndex], 0, 0, j - 1, LabelDataWidth);
                        lbls[lastIndex] = SubVector(lbls[lastIndex], 0, j - 1);
                    }
                }

                labelsEncoded = coded;
                labels = lbls;
                return data;
            }
        }

        public override IList<T[,]> Read(int count, int batchSize)
        {
            List<T[,]> data = new List<T[,]>();


            double cutOff = ((double)count) / TotalRecordCount;
            var rnd = new Random(DateTime.Now.Millisecond);

            using (FileStream fs = File.OpenRead(FilePath))
            using (var sr = new StreamReader(fs))
            {
                if (FirstLineIsHeaders)
                    sr.ReadLine();

                var dat = new T[batchSize, DataWidth];

                data.Add(dat);

                int lineNo = 0;
                string line;
                int j = 0;
                int newbatchSize = 0;
                for (line = sr.ReadLine(), lineNo = 0, j = 0;
                    !sr.EndOfStream && lineNo < count && !string.IsNullOrWhiteSpace(line);
                    line = sr.ReadLine())
                {
                    if (rnd.NextDouble() <= cutOff)
                    {
                        string[] lineParts = line.Split(FieldSeparator);
                        DataReader.CopyToTarget(dat, lineParts, j);

                        lineNo++;
                        j++;

                        if (j == dat.GetLength(0))
                        {
                            j = 0;
                            data.Add(dat);

                            newbatchSize = lineNo + batchSize < count ? batchSize : count - lineNo;
                            if (newbatchSize > 0)
                            {
                                dat = new T[newbatchSize, DataWidth];
                                data.Add(dat);
                            }
                        }
                    }
                }

                if (j < newbatchSize + 1)
                {
                    if (j - 1 < 1)
                    {
                        data.RemoveAt(data.Count - 1);
                    }
                    else
                    {
                        data[data.Count - 1] = SubMatrix(data[data.Count - 1], 0, 0, j - 1, DataWidth);
                    }
                }

                return data;
            }
        }

        //public override T[,] ReadWithLabels(int count, out T[,] labelsEncoded, out string[] labels, Func<T, T> sourceToTargetConverter)
        //{
        //    throw new NotImplementedException();
        //}

        //public override T[,] Read(int count, Func<T, T> sourceToTargetConverter)
        //{
        //    throw new NotImplementedException();
        //}

        //public override IList<T[,]> ReadWithLabels(int count, int batchSize, out IList<T[,]> labelsEncoded, out IList<string[]> labels, Func<T, T> sourceToTargetConverter)
        //{
        //    throw new NotImplementedException();
        //}

        //public override IList<T[,]> Read(int count, int batchSize, Func<T, T> sourceToTargetConverter)
        //{
        //    throw new NotImplementedException();
        //}
    }
}