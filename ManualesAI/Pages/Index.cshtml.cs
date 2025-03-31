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
            // No se requiere lógica en GET
        }

        // OnPostAsync no es necesario porque usaremos Fetch desde el navegador.
        // Si lo deseas, puedes dejarlo vacío o removerlo.
        public IActionResult OnPost()
        {
            return Page();
        }
    }
}

