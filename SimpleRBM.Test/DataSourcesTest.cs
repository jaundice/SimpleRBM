using System;
using System.Configuration;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SimpleRBM.Demo.Demo;

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
            var data = io.ReadTrainingData(0, 10, out labels);

            io.PrintToScreen(data, labels);
        }

        [TestMethod]
        public void TestKaggleTrainingData()
        {
            var io = new KaggleDataF(ConfigurationManager.AppSettings["KaggleTrainingData"],
                ConfigurationManager.AppSettings["KaggleTestData"]);

            int[] labels;
            var data = io.ReadTrainingData(0, 10, out labels);

            io.PrintToScreen(data, labels);
        }

        [TestMethod]
        public void TestMNist()
        {
            var provider = new MNistDataF(ConfigurationManager.AppSettings["FacesDirectory"]);
            string[] labels;
            var data = provider.ReadTrainingData(0, 10, out labels);

            provider.PrintToScreen(data, labels);
        }
    }
}
