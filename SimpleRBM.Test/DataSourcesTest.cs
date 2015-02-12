using System.Configuration;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SimpleRBM.Demo.Demo;
using SimpleRBM.Demo.IO;

namespace SimpleRBM.Test
{
    [TestClass]
    public class DataSourcesTest
    {
        [TestMethod]
        public void TestHandwrittenNumbers()
        {
            var io = new HandwrittenNumbersDataF("optdigits-tra.txt");

            int[] labels;
            float[,] labelsCoded;
            float[,] data = io.ReadTrainingData(0, 20, out labels, out labelsCoded);

            io.PrintToConsole(data, referenceLabels: labels, referenceLabelsCoded: labelsCoded);
        }

        [TestMethod]
        public void TestKaggleTrainingData()
        {
            var io = new KaggleDataF(ConfigurationManager.AppSettings["KaggleTrainingData"],
                ConfigurationManager.AppSettings["KaggleTestData"]);

            int[] labels;
            float[,] labelsCoded;
            float[,] data = io.ReadTrainingData(0, 20, out labels, out labelsCoded);

            io.PrintToConsole(data, referenceLabels: labels, referenceLabelsCoded: labelsCoded);
        }

        [TestMethod]
        public void TestMNist()
        {
            var provider = new FacesDataF(ConfigurationManager.AppSettings["FacesDirectory"]);
            string[] labels;
            float[,] labelsCoded;
            float[,] data = provider.ReadTrainingData(0, 20, out labels, out labelsCoded);

            provider.PrintToConsole(data, referenceLabels: labels, referenceLabelsCoded: labelsCoded);
        }
    }
}