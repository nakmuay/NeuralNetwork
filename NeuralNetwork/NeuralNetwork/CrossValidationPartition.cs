using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    public class CrossValidationPartition
    {
        private int[] indeces;
        private int[] permutedInces;
        private Random rand;

        private readonly int numberOfItems;
        private readonly int trainingSetSize;
        private readonly int testSetSize;
        private readonly int validationSetSize;

        private int[] trainingSet;
        private int[] testSet;
        private int[] validationSet;

        public CrossValidationPartition(int numberOfItems, double trainingSetProportion, double testSetProportion)
        {
            // Initialize random number generator
            rand = new Random();

            // Allocate indeces array
            this.indeces = new int[numberOfItems];
            this.permutedInces = new int[numberOfItems];
            for (int i = 0; i < numberOfItems; i++)
            {
                this.indeces[i] = i;
                this.permutedInces[i] = i;
            }

            // Randomize indeces
            Permute();

            this.numberOfItems = numberOfItems;
            initializeSetSizes(this.numberOfItems, trainingSetProportion, testSetProportion, out this.trainingSetSize, out this.testSetSize, out this.validationSetSize);
            intializeSetIndeces(out this.trainingSet, out this.testSet, out this.validationSet);
        }

        public CrossValidationPartition(int numberOfItems) : this(numberOfItems, 0.6, 0.3)
        {
        }

        #region properties

        public int[] TrainingSet
        {
            get
            {
                return trainingSet;
            }
        }

        public int[] TestSet
        {
            get
            {
                return testSet;
            }
        }

        public int[] ValidationSet
        {
            get
            {
                return validationSet;
            }
        }

        public int TrainingSetSize
        {
            get
            {
                return trainingSetSize;
            }
        }

        public int TestSetSize
        {
            get
            {
                return testSetSize;
            }
        }

        public int ValidationSetSize
        {
            get
            {
                return validationSetSize;
            }
        }

        public double TrainingSetProportion
        {
            get
            {
                return (double)trainingSetSize / (double)numberOfItems;
            }
        }

        public double TestSetProportion
        {
            get
            {
                return (double)testSetSize / (double)numberOfItems;
            }
        }

        public double ValidationSetProportion
        {
            get
            {
                return (double)validationSetSize / (double)numberOfItems;
            }
        }

        #endregion

        #region methods

        public void Permute()
        {
            // Declare temporary variables
            int firstRandomIndex;
            int secondRandomIndex;
            int tempIndex;

            // Permute indeces
            for (int i = 0; i < indeces.Length; i++)
            {
                firstRandomIndex = rand.Next(0, permutedInces.Length);
                tempIndex = permutedInces[firstRandomIndex];
                secondRandomIndex = rand.Next(0, permutedInces.Length);

                // Swap indeces
                permutedInces[firstRandomIndex] = permutedInces[secondRandomIndex];
                permutedInces[secondRandomIndex] = tempIndex;
            }

            intializeSetIndeces(out this.trainingSet, out this.testSet, out this.validationSet);
        }

        public void Write()
        {
            Console.Write("Training Set: ");
            for (int i = 0; i < TrainingSetSize; i++)
            {
                Console.Write("{0}, ", TrainingSet[i]);
            }
            Console.WriteLine();

            Console.Write("Test Set: ");
            for (int i = 0; i < TestSetSize; i++)
            {
                Console.Write("{0}, ", TestSet[i]);
            }
            Console.WriteLine();

            Console.Write("Validation Set: ");
            for (int i = 0; i < ValidationSetSize; i++)
            {
                Console.Write("{0}, ", ValidationSet[i]);
            }
            Console.WriteLine();
        }

        private void initializeSetSizes(int numberOfItems, double trainingSetProportion, double testSetProportion, 
                                        out int trainingSetSize, out int testSetSize, out int validationSetSize)
        {
            // Round training and test sets
            trainingSetSize     = (int)Math.Ceiling((double)numberOfItems * trainingSetProportion);
            testSetSize         = (int)Math.Floor((double)numberOfItems * testSetProportion);
            validationSetSize   = numberOfItems - (trainingSetSize + testSetSize);
        }

        private void intializeSetIndeces(out int[] trainingSet, out int[] testSet, out int[] validationSet)
        {
            // Get set indeces
            trainingSet     = getSubArray(permutedInces, 0, trainingSetSize);
            testSet         = getSubArray(permutedInces, trainingSetSize, testSetSize);
            validationSet   = getSubArray(permutedInces, testSetSize + trainingSetSize, validationSetSize);
        }

        // TODO [martin, 2017-04-06]: Refactor this using extension method and LINQ expression, e.g.: http://stackoverflow.com/questions/943635/getting-a-sub-array-from-an-existing-array
        private int[] getSubArray(int[] arr, int offset, int itemCount)
        {
            int[] subArray = new int[itemCount];
            for (int i = 0; i < itemCount; i++)
            {
                subArray[i] = arr[offset + i];
            }

            return subArray;
        }

        #endregion

    }

}
