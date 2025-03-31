using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.RazorPages;

namespace ManualesAI.Pages
{
    public class ConsultaModel : PageModel
    {
        [BindProperty]
        public string ConsultaTexto { get; set; }

        public void OnGet()
        {
            // No se requiere l�gica en GET
        }

        // OnPostAsync no es necesario porque usaremos Fetch desde el navegador.
        // Si lo deseas, puedes dejarlo vac�o o removerlo.
        public IActionResult OnPost()
        {
            return Page();
        }
    }
}

