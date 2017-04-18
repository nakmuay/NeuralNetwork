using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{

    public abstract class IdenfificationDataFactory
    {

        public abstract IdentificationDataSet GetData();

    }

    public class TestIdentificationDataFactory : IdenfificationDataFactory
    {

        private int numberOfExperiments;
        private int numberOfSamples;

        public TestIdentificationDataFactory(int numberOfExperiments, int numberOfSamples)
        {
            this.numberOfExperiments = numberOfExperiments;
            this.numberOfSamples = numberOfSamples;
        }

        public override IdentificationDataSet GetData()
        {
            // Declare return argument
            IdentificationDataSet data = new IdentificationDataSet();

            // Create random number generator for addition of noise
            Random rand = new Random();

            for (int i = 0; i < numberOfExperiments; i++)
            {
                // Create some training data
                double xMin = 0.0;
                double[][] input = new double[numberOfSamples][];
                double[][] output = new double[numberOfSamples][];
                for (int j = 0; j < numberOfSamples; j++)
                {
                    input[j] = new double[1];
                    input[j][0] = xMin + j / 100.0;

                    output[j] = new double[1];
                    output[j][0] = 1 / 2 * Math.Sin(2 * Math.PI * input[j][0]) + Math.Sin(2.5 * Math.PI * input[j][0])
                                      + Math.Sin(3.5 * Math.PI * input[j][0]) + 1 / 20 * Math.Sin(5 * Math.PI * input[j][0]);

                    output[j][0] *= 1 + (0.5 - rand.NextDouble()) / 2.0;
                }

                data.AddData(new IdentificationData(input, output));
            }

            return data;
        }
    }


    public class IdentificationDataSet
    {

        private List<IdentificationData> dataSet;
        private List<string> dataName;

        public IdentificationDataSet()
        {
            this.dataSet = new List<IdentificationData>();
            this.dataName = new List<string>();
        }

        #region properties

        public int Size
        {
            get
            {
                return dataSet.Count;
            }
        }

        public List<IdentificationData> Data
        {
            get
            {
                return dataSet;
            }
        }

        #endregion

        #region methods

        public void AddData(IdentificationData data, string name)
        {
            this.dataSet.Add(data);
            this.dataName.Add(name);
        }

        public void AddData(IdentificationData data)
        {
            AddData(data, String.Format("experiment_{0}", Size.ToString()));
        }

        public IdentificationDataSet GetSubset(int[] index)
        {
            int numItems = index.Length;
            IdentificationDataSet subset = new IdentificationDataSet();

            for (int i = 0; i < numItems; i++)
            {
                subset.AddData(this.dataSet[index[i]]);
            }

            return subset;
        }

        public bool TrySerialize(string pathname)
        {
            string fileEnding = ".csv";
            for (int i = 0; i < this.Size; i++)
            {
                string filename = Path.Combine(pathname, dataName[i] + fileEnding);
                dataSet[i].TrySerialize(filename);
            }

            return true;
        }

        #endregion

    }

    public class IdentificationData
    {

        public double[][] inputData;
        public double[][] outputData;

        public IdentificationData(double[][] inputData)
        {
            this.inputData = inputData;
            this.outputData = new double[inputData.Length][];
        }

        public IdentificationData(double[][] inputData, double[][] outputData) : this(inputData)
        {
            this.outputData = outputData;
        }

        #region properties

        public double[][] InputData
        {
            get
            {
                return inputData;
            }
        }

        public int NumInputVariables
        {
            get
            {
                return inputData[0].Length;
            }
        }

        public double[][] OutputData
        {
            get
            {
                return outputData;
            }
        }

        public int NumOutputVariables
        {
            get
            {
                return outputData[0].Length;
            }
        }

        public int NumSamples
        {
            get
            {
                return inputData.Length;
            }
        }

        #endregion properties

        #region public methods

        public bool TrySerialize(string filename)
        {
            using (System.IO.StreamWriter file = new FormattingStreamWriter(@filename, System.Globalization.CultureInfo.InvariantCulture))
            {
                for (int i = 0; i < NumSamples; i++)
                {
                    file.WriteLine("{0}\t{1}", InputData[i][0], OutputData[i][0]);
                }
            }

            return true;
        }

        #endregion

    }

    public class FormattingStreamWriter : StreamWriter
    {
        private readonly IFormatProvider formatProvider;

        public FormattingStreamWriter(string path, IFormatProvider formatProvider)
            : base(path)
        {
            this.formatProvider = formatProvider;
        }
        public override IFormatProvider FormatProvider
        {
            get
            {
                return this.formatProvider;
            }
        }
    }
}
