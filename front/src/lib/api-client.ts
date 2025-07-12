// src/api-client.ts

// --- Définition des types basés sur votre schéma Swagger ---

/**
 * Paramètres pour la génération de texte.
 * Toutes les propriétés sont optionnelles pour plus de flexibilité.
 */
export interface GenerationParams {
  n?: number;
  prompt?: string;
  top_p?: number;
  top_k?: number;
  max_length?: number;
  temperature?: number;
}

/**
 * Détails d'un modèle spécifique.
 */
export interface ModelDetails {
  name: string;
  version: string;
  description: string;
  device: 'cpu' | 'cuda' | string; // 'cpu', 'cuda', ou autre
  tokenizer: {
    description: string;
    name: string;
    version: string;
    vocab: { [key: string]: string };
    vocab_size: number;
  };
}

/**
 * Résultat d'une requête de génération standard (non-stream).
 */
export interface GenerationResult {
  id_: string;
  model_id: string;
  time_elapsed: number;
  avg_time: number;
  nrps: number;
  ntps: number;
  results: string[];
  uniques: string[];
}

/**
 * Callbacks pour la gestion du flux de streaming.
 */
export interface StreamCallbacks {
  /**
   * Appelé à chaque fois qu'un nouveau token (morceau de texte) est reçu.
   * @param token Le token reçu du serveur.
   */
  onToken: (token: string) => void;
  /**
   * Appelé lorsque le flux est terminé avec succès.
   */
  onEnd: () => void;
  /**
   * Appelé si une erreur survient pendant le streaming.
   * @param error L'objet d'erreur.
   */
  onError: (error: Error) => void;
}


// --- Classe du Client API ---

export class ApiClient {
  private readonly baseUrl: string;

  /**
   * Crée une nouvelle instance du client API.
   * @param baseUrl L'URL de base de votre API (ex: http://localhost:8000/api/v1)
   */
  constructor(baseUrl: string) {
    // S'assure que l'URL ne se termine pas par un '/' pour éviter les doubles slashes
    this.baseUrl = baseUrl.endsWith('/') ? baseUrl.slice(0, -1) : baseUrl;
  }

  /**
   * Gère les réponses HTTP et les erreurs potentielles.
   * @param response La réponse de l'API fetch.
   */
  private async handleResponse<T>(response: Response): Promise<T> {
    console.log("handleGenerate a été appelé !"); // <--- AJOUTEZ CETTE LIGNE
    if (!response.ok) {
      // Tente de parser le corps de l'erreur s'il existe
      const errorBody = await response.json().catch(() => ({ detail: 'An unknown error occurred.' }));
      const errorMessage = errorBody.detail || `HTTP error! Status: ${response.status}`;
      throw new Error(JSON.stringify(errorMessage));
    }
    return response.json() as Promise<T>;
  }


  /**
   * Route : GET /
   * Récupère la liste des identifiants des modèles disponibles.
   * @returns Une promesse qui se résout avec un tableau de noms de modèles.
   */
  public async listModels(): Promise<string[]> {
    const response = await fetch(`${this.baseUrl}/`);
    return this.handleResponse<string[]>(response);
  }

  /**
   * Route : GET /{id_}
   * Récupère les détails d'un modèle spécifique.
   * @param modelId L'identifiant du modèle (ex: "communes").
   * @returns Une promesse qui se résout avec les détails du modèle.
   */
  public async getModelDetails(modelId: string): Promise<ModelDetails> {
    const response = await fetch(`${this.baseUrl}/${modelId}`);
    return this.handleResponse<ModelDetails>(response);
  }

  /**
   * Route : POST /{id_}/generate
   * Génère des résultats complets à partir d'un modèle.
   * @param modelId L'identifiant du modèle.
   * @param params Les paramètres d'inférence.
   * @returns Une promesse qui se résout avec les résultats de la génération.
   */
  public async generate(modelId: string, params: GenerationParams): Promise<GenerationResult> {
    const response = await fetch(`${this.baseUrl}/${modelId}/generate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
      body: JSON.stringify(params),
    });
    return this.handleResponse<GenerationResult>(response);
  }

  /**
   * Route : POST /{id_}/stream
   * Génère des résultats token par token en utilisant les Server-Sent Events (SSE).
   * @param modelId L'identifiant du modèle.
   * @param params Les paramètres d'inférence.
   * @param callbacks Un objet contenant les fonctions onToken, onEnd, et onError.
   */
  public async stream(modelId: string, params: GenerationParams, callbacks: StreamCallbacks): Promise<void> {
    try {
      const response = await fetch(`${this.baseUrl}/${modelId}/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'text/event-stream',
        },
        body: JSON.stringify(params),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }

      if (!response.body) {
          throw new Error('Response body is null.');
      }

      // Utilisation de TextDecoderStream pour gérer correctement les encodages (ex: UTF-8)
      const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();

      while (true) {
        const { done, value } = await reader.read();

        if (done) {
          callbacks.onEnd();
          break;
        }

        // Un chunk peut contenir plusieurs événements, on les traite ligne par ligne
        const lines = value.split('\n').filter(line => line.trim() !== '');

        for (const line of lines) {
            if (line.startsWith('data: ')) {
                // Extrait le contenu après "data: "
                const data = line.substring(6).trim();
                if (data === 'Stream finished') {
                    // C'est un message de fin personnalisé, on peut l'ignorer ici car `done` ci-dessus est plus fiable.
                    continue;
                }
                callbacks.onToken(data);
            } else if (line.startsWith('event: end')) {
                // L'API envoie un événement 'end' pour signaler la fin.
                // La boucle se terminera quand `done` sera `true`, mais on pourrait aussi appeler onEnd() ici.
            }
        }
      }

    } catch (error) {
      callbacks.onError(error instanceof Error ? error : new Error(String(error)));
    }
  }
}