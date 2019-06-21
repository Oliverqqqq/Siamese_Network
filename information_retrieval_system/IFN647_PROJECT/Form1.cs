using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Diagnostics;
using Lucene.Net.Analysis; // for Analyser
using Lucene.Net.Documents; // for Document
using Lucene.Net.Index; //for Index Writer
using Lucene.Net.Store; //for Directory
using Lucene.Net.Analysis.Standard;
using Lucene.Net.Search; // for IndexSearcher
using Lucene.Net.QueryParsers;  // for QueryParser
using Lucene.Net.Analysis.Snowball;
using System.Collections.Generic;

namespace IFN647_PROJECT
{
    public partial class indexForm1 : Form
    {
        public static float titleboost;
        public static float authorboost;
        
        
        public indexForm1()
        {
            InitializeComponent();
        }

        

        

        private void indexbutton1_Click(object sender, EventArgs e)
        {
            indexBrowserDialog1.ShowDialog();
            indexBox1.Text = indexBrowserDialog1.SelectedPath;
        }

        private void filebutton2_Click(object sender, EventArgs e)
        {
            
            resourceBrowserDialog1.ShowDialog();
            collectBox2.Text = resourceBrowserDialog1.SelectedPath;
        }

        private void helpbutton_Click(object sender, EventArgs e)
        {
            helpForm help = new helpForm();
            help.Show();
        }

        private void conformbutton1_Click(object sender, EventArgs e)
        {
            if (indexBox1.Text != "" && collectBox2.Text != "")
            {
                Stopwatch stopwatch = new Stopwatch();
                LuceneAPP luceneAPP = new LuceneAPP();
                stopwatch.Restart(); //record start time
                titleboost = (float) title.Value;
                authorboost = (float)author.Value;               
                luceneAPP.OpenIndex(indexBox1.Text);
                luceneAPP.CreatAnalyzer_writer();                
                luceneAPP.Read_directory(collectBox2.Text);
                LuceneAPP.title_field.Boost = titleboost;
                LuceneAPP.author_field.Boost = authorboost;
                stopwatch.Stop();//record stop time
                luceneAPP.CleanUpIndex();
                var time = stopwatch.Elapsed;//execute time
                MessageBox.Show("Index completed!!\nExecute Time: " + time + " seconds.");

            }
            else
            {
                MessageBox.Show("please set your path");
            }
        }

        private void SEARCHbutton_Click(object sender, EventArgs e)
        {
            searchingForm searching = new searchingForm();
            searching.Show();
            this.Hide();
        }
    }
}
