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

                data.AddData(new IdentificationData(input, output, String.Format("experiment_{0}", i)));
            }

            return data;
        }
    }


    public class IdentificationDataSet
    {

        private List<IdentificationData> dataSet;

        public IdentificationDataSet()
        {
            this.dataSet = new List<IdentificationData>();
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

        public void AddData(IdentificationData data)
        {
            this.dataSet.Add(data);
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

        public bool TrySerialize(System.IO.StreamWriter writer, string delimiter=",")
        {
            for (int i = 0; i < this.Size; i++)
            {
                dataSet[i].TrySerialize(writer, delimiter);
            }

            return true;
        }

        #endregion

    }

    public class IdentificationData
    {
        // Declare internal fields
        private readonly double[][] inputData;
        private readonly double[][] outputData;

        private string[] inputName;
        private string[] outputName;

        private string name;

        public IdentificationData(double[][] inputData, string name= "")
        {
            this.inputData = inputData;
            this.outputData = new double[inputData.Length][];
            this.name = name;

            // Initialize input variable names
            inputName = new string[NumInputVariables];
            for (int i = 0; i < NumInputVariables; i++)
            {
                inputName[i] = String.Format("input_var_{0}", i);
            }
        }

        public IdentificationData(double[][] inputData, double[][] outputData, string name="") : this(inputData, name)
        {
            this.outputData = outputData;

            // Initialize output variable names
            outputName = new string[NumOutputVariables];
            for (int i = 0; i < NumOutputVariables; i++)
            {
                outputName[i] = String.Format("output_var_{0}", i);
            }
        }

        #region properties

        public double[][] InputData
        {
            get
            {
                return inputData;
            }
        }

        public string[] InputName
        {
            get
            {
                return inputName;
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

        public string[] OutputName
        {
            get
            {
                return outputName;
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

        public string Name
        {
            get
            {
                return this.name;
            }
            set
            {
                this.name = value;
            }
        }

        #endregion properties

        #region public methods

        public bool TrySerialize(System.IO.StreamWriter writer, string delimiter=",")
        {
            writer.WriteLine(String.Format("Name:{0}", this.Name));
            writer.WriteLine(String.Format("Number_of_samples:{0}", this.NumSamples));

            // Write header
            for (int i = 0; i < NumInputVariables; i++)
            {
                writer.Write(String.Format("input_{0}:{1}{2}", i, InputName[i], delimiter));
            }

            for (int i = 0; i < NumOutputVariables; i++)
            {
                writer.Write(String.Format("output_{0}:{1}{2}", i, OutputName[i], delimiter));
            }
            writer.WriteLine();

            // Write data
            for (int i = 0; i < NumSamples; i++)
            {
                // Write input data
                for (int j = 0; j < NumInputVariables; j++)
                {
                    writer.Write("{0}{1}", InputData[i][0], delimiter);
                }

                // Write output data
                for (int j = 0; j < NumOutputVariables; j++)
                {
                    writer.Write("{0}{1}", OutputData[i][0], delimiter);
                }
                writer.WriteLine();
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
