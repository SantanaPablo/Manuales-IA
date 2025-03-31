using Microsoft.AspNetCore.Mvc;
using System.Net;
using System.Net.Http;
using System.Threading.Tasks;

namespace ManualesAI.Services
{
    [ApiController]
    [Route("api/[controller]")]
    public class ProxyController : ControllerBase
    {
        private readonly IHttpClientFactory _httpClientFactory;

        public ProxyController(IHttpClientFactory httpClientFactory)
        {
            _httpClientFactory = httpClientFactory;
        }

        [HttpGet]
        public async Task<IActionResult> Get([FromQuery] string pregunta)
        {
            if (string.IsNullOrWhiteSpace(pregunta))
            {
                return BadRequest("La pregunta es requerida.");
            }

            // Construir la URL al backend Python. Aquí aún se usa 127.0.0.1 porque es el servidor.
            var backendUrl = $"http://127.0.0.1:8000/buscar/?pregunta={WebUtility.UrlEncode(pregunta)}";

            // Crear el HttpClient
            var client = _httpClientFactory.CreateClient();

            // Realizar la petición al backend Python, pidiendo que la respuesta se lea de forma progresiva
            var backendResponse = await client.GetAsync(backendUrl, HttpCompletionOption.ResponseHeadersRead);

            if (!backendResponse.IsSuccessStatusCode)
            {
                return StatusCode((int)backendResponse.StatusCode, "Error en el backend.");
            }

            // Configurar el tipo de contenido según lo que envíe el backend (en este caso, "application/json")
            Response.ContentType = "application/json";

            // Stream la respuesta del backend directamente a la respuesta del cliente
            using (var responseStream = await backendResponse.Content.ReadAsStreamAsync())
            {
                await responseStream.CopyToAsync(Response.Body);
            }

            return new EmptyResult();
        }
    }
}
