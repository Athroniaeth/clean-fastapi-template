<script lang="ts">
    // --- Imports SvelteKit & Composants ---
    import {
        Footer, FooterCopyright, GradientButton, Input, Label, Navbar, NavBrand, NavHamburger, NavLi, NavUl, Spinner
    } from "flowbite-svelte";
    import { AnnotationSolid } from "flowbite-svelte-icons";
    import ModelRadio from "./ModelRadio.svelte";
    import ParameterSlider from "./ParameterSlider.svelte";

    // --- Imports pour l'API ---
    import { ApiClient, type GenerationParams } from '$lib/api-client'; // Import du client

    // --- Instance du client API ---
    const PUBLIC_API_BASE_URL = "http://localhost:8000/api/v1"; // URL de l'API publique
    const apiClient = new ApiClient(PUBLIC_API_BASE_URL);

    // --- État (State) de l'application avec les runes Svelte 5 ---

    // Paramètres de génération
    let temperature = $state(0.6);
    let topP = $state(0.95);
    let topK = $state(0);
    let maxTokens = $state(30);
    let nResponses = $state(5);
    let prompt = $state('');

    // Données de l'API et état de l'UI
    let models = $state<string[]>([]);
    let selectedModel = $state('');
    let results = $state<string[]>([]);
    let isLoading = $state(false);
    let error = $state<string | null>(null);

    // MODIFICATION 1: Ajout de l'état pour l'historique
    type HistoryEntry = { prompt: string, results: string[] };
    let history = $state<HistoryEntry[]>([]);

    // --- Chargement des données initiales ---
    (async () => {
        try {
            const modelList = await apiClient.listModels();
            models = modelList;
            if (modelList.length > 0) {
                selectedModel = modelList[0];
            }
        } catch (e: any) {
            error = e.message;
        }
    })();


    // MODIFICATION 2: Fonction de génération mise à jour
    // --- Fonction pour lancer la génération (version corrigée) ---
async function handleGenerate() {
    if (!selectedModel) {
        error = "Veuillez sélectionner un modèle.";
        return;
    }

    isLoading = true;
    error = null;

    // On initialise le tableau des résultats avec le prompt.
    // Cette assignation initiale met à jour l'interface une première fois.
    results = Array(nResponses).fill(prompt);

    const params: GenerationParams = {
        prompt: prompt,
        n: 1, // On demande un stream par résultat
        temperature: temperature,
        top_p: topP,
        top_k: topK,
        max_length: maxTokens,
    };

    const streamPromises = Array.from({ length: nResponses }).map((_, i) =>
        apiClient.stream(selectedModel, params, {
            onToken: (token) => {
                // 1. On met à jour la valeur dans notre tableau en mémoire
                results[i] += token;

                // 2. LA CORRECTION MAGIQUE :
                // On assigne une nouvelle copie du tableau à la variable réactive 'results'.
                // Svelte voit une nouvelle assignation et est OBLIGÉ de redessiner le {#each} block.
                results = [...results];
            },
            onEnd: () => {
                console.log(`Stream ${i + 1}/${nResponses} terminé.`);
            },
            onError: (e) => {
                error = `Erreur sur le stream ${i + 1}: ${e.message}`;
            },
        })
    );

    try {
        await Promise.all(streamPromises);
        // À la fin, on sauvegarde une copie du tableau final (qui a été correctement mis à jour)
        history.unshift({ prompt: prompt || '(no prompt)', results: [...results] });

    } catch (e: any) {
        error = e.message;
    } finally {
        isLoading = false;
    }
}

</script>

<!-- src/routes/+page.svelte -->

<div class="container">
    <div class="header pl-15 pr-40">
        <!-- Navbar reste identique -->
        <Navbar class="px-2 sm:px-4 py-2.5 dark:bg-gray-900">
            <NavBrand href="/">
                <img src="https://flowbite.com/docs/images/logo.svg" class="mr-3 h-6 sm:h-9" alt="Rename Logo"/>
                <span class="self-center whitespace-nowrap text-xl font-semibold text-blue-950 dark:text-white">Rename</span>
            </NavBrand>
            <NavHamburger class="text-white hover:bg-blue-800"/>
            <NavUl>
                <NavLi href="/">Accueil</NavLi>
                <NavLi href="/">À Propos</NavLi>
                <NavLi href="/">Services</NavLi>
                <NavLi href="/">Contact</NavLi>
            </NavUl>
        </Navbar>
    </div>

    <!-- Section des modèles, maintenant dynamique -->
    <div class="models" style="padding-top: 15px;">
        {#if models.length > 0}
            {#each models as model}
                <ModelRadio bind:group={selectedModel} value={model} description={`Modèle: ${model}`}></ModelRadio>
            {/each}
        {:else}
            <p>Chargement des modèles...</p>
        {/if}
    </div>

    <div class="panel" style="padding-top: 15px;padding-left: 30px;padding-right: 30px;">
        <div class="mb-6">
            <Label for="default-input" class="mb-2 block">Texte de départ (Prompt)</Label>
            <!-- Lier l'input à la variable `prompt` -->
            <Input id="default-input" placeholder="Ex: 'Saint-Jean-de-'" bind:value={prompt}/>
        </div>
        <!-- Affichage des résultats de la génération -->
        <div class="max-h-75 overflow-y-auto border border-gray-200 p-4">
            <ul class="space-y-2">
                {#if results.length === 0 && !isLoading}
                    <li class="text-gray-400">Les résultats apparaîtront ici...</li>
                {/if}
                {#each results as resultItem}
                    <li class="px-2 py-1 bg-gray-50 rounded min-h-[1.5em]">{resultItem}</li>
                {/each}
            </ul>
        </div>
        <!-- Boutons de génération conditionnels -->
        <!-- Boutons de génération conditionnels -->
        <div class="flex mt-6">
            {#if isLoading}
                <GradientButton class="mr-6" color="blue" disabled>
                    <Spinner class="me-3" size="4" color="gray"/>
                    Génération en cours...
                </GradientButton>
            {:else}
                <!-- CORRECTION: Utilisation de `on:click` au lieu de `onclick` -->
                <GradientButton class="mr-6" color="blue" onclick={handleGenerate}>
                    Générer
                    <AnnotationSolid class="ms-2 h-5 w-5"/>
                </GradientButton>
            {/if}
        </div>
        <!-- Affichage d'erreur -->
        {#if error}
            <p class="mt-4 text-red-600">{error}</p>
        {/if}
    </div>

    <div class="parameters p-5">
        <!-- Sliders de paramètres (inchangés, car déjà liés aux variables d'état) -->
        <ParameterSlider id="temperature-range" label="Temperature" min={0} max={3} step={0.01} bind:value={temperature}/>
        <ParameterSlider id="top-p-range" label="Top P" min={0} max={1} step={0.01} bind:value={topP}/>
        <ParameterSlider id="top-k-range" label="Top K" min={0} max={100} step={1} bind:value={topK}/>
        <ParameterSlider id="max-tokens-range" label="Max Tokens" min={1} max={1000} step={1} bind:value={maxTokens}/>
        <ParameterSlider id="n-responses-range" label="N Responses" min={1} max={10} step={1} bind:value={nResponses}/>
    </div>
        <!-- MODIFICATION 3: Section "history" rendue dynamique -->
    <div class="history">
        <div class="max-h-125 overflow-y-auto border border-gray-200 p-4 ">
            {#if history.length > 0}
                <ul>
                    <!-- On boucle sur notre nouvel état 'history' -->
                    {#each history as entry, i (entry.prompt + i)}
                        <div class="mb-4">
                            <li class="px-2 py-1 bg-gray-50 rounded font-semibold text-gray-800">
                                Prompt: "{entry.prompt}"
                            </li>
                            <ul class="ml-4 list-disc list-inside p-1 text-sm text-gray-600">
                                <!-- Affiche les résultats de cette entrée d'historique -->
                                {#each entry.results as resultItem}
                                    <li>{resultItem}</li>
                                {/each}
                            </ul>
                        </div>
                    {/each}
                </ul>
            {:else}
                <p class="text-gray-400">L'historique des générations apparaîtra ici.</p>
            {/if}
        </div>
    </div>
    <div class="footer">
        <Footer>
            <FooterCopyright style="color:dodgerblue" href="/" by="Athroniaeth" year={2025}>
                All rights reserved.
            </FooterCopyright>
            <!--
            <FooterLinkGroup class="mt-3 flex flex-wrap items-center text-sm  sm:mt-0 ">

                <FooterLink style="color:dodgerblue" href="/">About</FooterLink>
                <FooterLink style="color:dodgerblue" href="/">About</FooterLink>
                <FooterLink style="color:dodgerblue" href="/">Privacy Policy</FooterLink>
                <FooterLink style="color:dodgerblue" href="/">Licensing</FooterLink>
                <FooterLink style="color:dodgerblue" href="/">Contact</FooterLink>

            </FooterLinkGroup>
            -->
        </Footer>
    </div>
</div>

<style>
    /*
      Règle globale pour s'assurer que le layout prend 100% de l'écran.
      On retire les marges par défaut du navigateur.
    */
    :global(html, body) {
        margin: 0;
        padding: 0;
        height: 100%;
        font-family: sans-serif;
    }

    .container {
        /* Dimensions pour occuper tout l'écran */
        height: 100vh; /* 100% de la hauteur de la fenêtre (viewport height) */
        width: 100vw; /* 100% de la largeur de la fenêtre (viewport width) */

        /* Activation de la grille CSS */
        display: grid;

        /* Définition des colonnes (6 colonnes) */
        /* Les valeurs sont une interprétation pour correspondre à l'image */
        grid-template-columns: 0.5fr 4fr 0.5fr 9fr 0.5fr 5fr 0.5fr 4fr 0.5fr;

        /* Définition des lignes (5 lignes) */
        grid-template-rows: 1fr 2fr 2fr 2fr 1fr;

        /* C'est ici la magie : on nomme et on place les zones */
        /* Le "." représente une cellule de grille vide */
        grid-template-areas:
    "header header header header header header header header header"
    ". models . panel . parameters . history ."
    ". models . panel . parameters . history ."
    ". models . panel . parameters . history ."
    "footer footer footer footer footer footer footer footer footer";

        /* Un peu d'espacement pour mieux visualiser les zones */
        /* gap: 5px; */
        background-color: #f2f5ff; /* Couleur de fond pour les espaces (gap) */
    }


    .container {
        display: grid;
        grid-template-columns: 0.75fr 4fr 0.75fr 10fr 0.75fr 4fr 0.75fr 6fr 0.75fr;
        grid-template-rows: 0.5fr 0.25fr 2fr 2fr 0.25fr 0.5fr;
        gap: 0px 0px;
        grid-auto-flow: row;
        grid-template-areas:
    "header header header header header header header header header"
    ". . . . . . . . ."
    ". models . panel . parameters . history ."
    ". models . panel . parameters . history ."
    ". . . . . . . . ."
    "footer footer footer footer footer footer footer footer footer";
    }

    .history {
        grid-area: history;
    }

    .parameters {
        grid-area: parameters;
    }

    .panel {
        grid-area: panel;
    }

    .models {
        grid-area: models;
    }

    .header {
        grid-area: header;
    }

    .footer {
        grid-area: footer;
    }


    /* --- Style visuel pour la démonstration --- */
    .container > div {
        /* display: grid; */
        /* place-items: center; /* Centre le texte */
        /* color: black; */
        font-size: 1rem;
        font-weight: normal;
    }

    .header {
        background-color: white;
    }

    .models {
        background-color: white;
        place-items: center;
    }

    .panel {
        background-color: white;
    }

    .parameters {
        background-color: white;
    }

    .history {
        background-color: white;
    }

    .footer {
        background-color: whitesmoke;
    }
</style>
