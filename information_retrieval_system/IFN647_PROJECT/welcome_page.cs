using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace IFN647_PROJECT
{
    public partial class welcome_page : Form
    {
        public welcome_page()
        {
            InitializeComponent();
        }

        private void beginbutton1_Click(object sender, EventArgs e)
        {
            indexForm1 form = new indexForm1();

            this.Hide();
            form.ShowDialog();
        }
    }
}
