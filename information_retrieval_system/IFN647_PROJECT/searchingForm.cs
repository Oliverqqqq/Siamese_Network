using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Windows.Forms;
using Lucene.Net.Analysis; // for Analyser
using Lucene.Net.Documents; // for Document
using Lucene.Net.Index; //for Index Writer
using Lucene.Net.Store; //for Directory
using Lucene.Net.Analysis.Standard;
using Lucene.Net.Search; // for IndexSearcher
using Lucene.Net.QueryParsers;  // for QueryParser
using System.Diagnostics;
using System.IO;
using Syn.WordNet;





namespace IFN647_PROJECT
{
    public partial class searchingForm : Form
    {
        int currentpage = 1;
        int totalpage;
        
        int pagesize = 10;
        public static int current_index;
        LuceneAPP lucene = new LuceneAPP();
        
        TopDocs results;
        Stopwatch stopwatch = new Stopwatch();
        ScoreDoc[] score ;
        string input = "";
        WordNetEngine wordNet;
        string weighted_text;
        string[] des_array;







        public searchingForm()
        {
            InitializeComponent();
            
        }

        public void button1_Click(object sender, EventArgs e)
        {
            querydiaplay.Clear();
            if (searchBox1.Text == "")
            {
                MessageBox.Show("Please Enter Your Query");
            }

            else
            {
                if (PhrasecheckBox1.Checked)
                {

                    input = ("\"" + searchBox1.Text + "\"");
                    querydiaplay.AppendText("Searching for\n");
                    querydiaplay.AppendText(input);


                }
               if (preprocessing.Checked)//if user choose preprocessing, application will tokenization, remove stop word and extract stemming
                  {
                        input = lucene.preprocessing(searchBox1.Text);
                        querydiaplay.AppendText("Searching for\n");
                        querydiaplay.AppendText(input);

                 }

                    if (synonym.Checked)
                    {
                        var directory = System.IO.Directory.GetCurrentDirectory(); // Set WordNet directory
                        wordNet = new WordNetEngine();  // Initiate WordNet object
                        wordNet.LoadFromDirectory(directory);
                        MessageBox.Show("loading wornet successful");
                        string input_text = searchBox1.Text;
                        int letterindex;
                        /*
                        string temp_s = "";
                        bool quotesOpened = false;
                        for (letterindex = 0; letterindex < input_text.Length; letterindex++)
                        {
                            if (input_text[letterindex] == '"')
                            {
                                quotesOpened = !quotesOpened;
                            }
                            else
                            {
                                if (!quotesOpened && input_text[letterindex] == '"')
                                    temp_s = temp_s + input_text[letterindex];
                            }

                        }
                        */
                        string[] separators = { ",", ".", "!", "?", ";", ":", "-", " ", "\n", "\"", "'" };
                        string[] input_array = input_text.Split(separators, StringSplitOptions.RemoveEmptyEntries);
                        for (int i = 0; i < input_array.Length; i++)
                        {

                            weighted_text += input_array[i] + "^5";
                        }


                        input = weighted_text + Expansion_query(input_array);
                        querydiaplay.AppendText("Searching for\n");
                        querydiaplay.AppendText(input);

                    }
                if (!preprocessing.Checked&&!synonym.Checked&&!PhrasecheckBox1.Checked)
                {
                    input = searchBox1.Text;
                    querydiaplay.AppendText("Searching for\n");
                    querydiaplay.AppendText(input);
                }
                    


                    stopwatch.Restart();
                    
                    lucene.CreatParser();
                    results = lucene.Searching(input);
                    if (results.TotalHits % 10 == 0)
                    {
                        totalpage = results.TotalHits / 10;
                    }
                    else
                    {
                        totalpage = results.TotalHits / 10 + 1;
                    }
                    searchBypage(results, currentpage, pagesize);
                    stopwatch.Stop();
                    label3.Text = (currentpage + " / " + totalpage);
                    textBox1.Text = results.TotalHits.ToString();
                    textBox2.Text = stopwatch.Elapsed.ToString();
                }
            }
        
        public string Expansion_query(string[] query_token)
        {
            string expandquery = "";
            foreach (string t in query_token)
            {
                var synSetlist = wordNet.GetSynSets(t);
                if (synSetlist.Count !=0)
                {
                    foreach (var synSet in synSetlist)
                    {
                        foreach (var word in synSet.Words)
                        {
                            if (word != t)
                            {
                                expandquery+=word;
                            }

                        }
                    }

                }
            }
            return expandquery.Replace('_',' ');
        }
        public void searchBypage(TopDocs results, int now_page,int size)
        {
            
           DataTable dt = new DataTable();
            if (results.TotalHits == 0)
            {
                dt.Clear();
                dataGridView1.DataSource = dt;
                MessageBox.Show("no results");
            }
            else
            {
                dt.Columns.Add("Rank", typeof(int));
                dt.Columns.Add("Title", typeof(string));
                dt.Columns.Add("Author", typeof(string));
                dt.Columns.Add("BIbli Info", typeof(string));
                dt.Columns.Add("Abstract", typeof(string));

                
                int start = (now_page - 1) * size;
                int end = (now_page * size);

                score = results.ScoreDocs;

                for (int i = start; i < end&&i<results.TotalHits ; i++)
                {
                    Document doc = LuceneAPP.searcher.Doc(score[i].Doc);
                    dt.Rows.Add(i + 1, doc.Get("TItle").ToString(), doc.Get("Author").ToString()
                        , doc.Get("BiblInfo").ToString(), doc.Get("Abstract").Split('.').First());
                    dataGridView1.DataSource = dt;
                }
            }
        }

        private void nextbutton1_Click(object sender, EventArgs e)
        {
            if (currentpage < totalpage)
            {
                currentpage++;
                searchBypage(results, currentpage, pagesize);
                label3.Text = (currentpage + " / " + totalpage);
            }
        }

        private void BACKbutton1_Click(object sender, EventArgs e)
        {
            if (currentpage != 1)
            {
                currentpage--;
                searchBypage(results, currentpage, pagesize);
                label3.Text = (currentpage + " / " + totalpage);
            }
        }

        private void viewbutton1_Click_1(object sender, EventArgs e)
        {

            current_index = dataGridView1.CurrentRow.Index + ((currentpage - 1) * 10);
            Document doc = LuceneAPP.searcher.Doc(score[current_index].Doc);
            MessageBox.Show("Abstract:\n"+doc.Get("Abstract").ToString());
        }

        private void dataGridView1_Click(object sender, EventArgs e)
        {
            current_index = dataGridView1.CurrentRow.Index + ((currentpage - 1) * 10);
            Document doc = LuceneAPP.searcher.Doc(score[current_index].Doc);
            MessageBox.Show("Abstract:\n" + doc.Get("Abstract").ToString());
            

        }

        private void Savebutton_Click(object sender, EventArgs e)
        {
            if (textBox3.Text == "")
            {
                MessageBox.Show("Please Set Your Topic ID");
            }
            else
            {
                saveFileDialog1.Filter = "txt files(*.txt)|*.txt";
                saveFileDialog1.RestoreDirectory = true;
                saveFileDialog1.OverwritePrompt = false;
                if (saveFileDialog1.ShowDialog() == DialogResult.OK)
                {
                    string savePath = saveFileDialog1.FileName.ToString();
                    string topicID = textBox3.Text;
                    StreamWriter writer = File.AppendText(savePath);
                    score = results.ScoreDocs;

                    int score_length = score.Length;

                    for (int i = 0; i < score_length; i++)
                    {
                        Document doc = LuceneAPP.searcher.Doc(score[i].Doc);
                        writer.WriteLine(textBox3.Text + "\t"+"Q0"+"\t" + doc.Get("ID").ToString() + "\t" + (i + 1) + "\t" + score[i].Score + "\t"+"StudentNumber");
                    }

                    writer.Flush();
                    writer.Close();
                }
            }
        }

        private void load_infor_Click(object sender, EventArgs e)
        {
            openFileDialog1.ShowDialog();
            if (openFileDialog1.ShowDialog() == DialogResult.OK)
            {
                System.IO.StreamReader reader = new System.IO.StreamReader(openFileDialog1.FileName);
                string informationNeeds = reader.ReadToEnd();
                string[] delims = { ".I", ".D" };
                string[] infor_splite = informationNeeds.Split(delims, StringSplitOptions.RemoveEmptyEntries);
                DataTable dt2 = new DataTable();
                dt2.Columns.Add("ID", typeof(int));
                dt2.Columns.Add("Describe", typeof(string));
                List<string> ID = new List<string>();
                List<string> describe = new List<string>();
                for (int i = 0; i < infor_splite.Length; i++)
                {
                    if (i % 2 == 0)
                        ID.Add(infor_splite[i]);
                    else
                        describe.Add(infor_splite[i]);
                }
                string[] id_array=ID.ToArray();
                 des_array = describe.ToArray();
                for (int i = 0; i < 5; i++)
                {
                    dt2.Rows.Add(id_array[i],des_array[i]);
                }
                dataGridView2.DataSource = dt2;




            }

        }

        private void dataGridView2_CellContentClick(object sender, DataGridViewCellEventArgs e)
        {
            searchBox1.Text = des_array[dataGridView2.CurrentRow.Index];
        }
    }
}

