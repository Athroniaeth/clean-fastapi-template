<script>
    import ModelRadio from "./ModelRadio.svelte";
    import {GradientButton,Spinner, Button, Input, Label} from "flowbite-svelte";
  import { AnnotationSolid, ArrowRightOutline, CartSolid } from "flowbite-svelte-icons";
    let selectedModel = 'communes:v1';
    let items = Array.from({length: 30}, (_, i) => `Élément ${i + 1}`);
</script>

<!-- src/routes/+page.svelte -->

<div class="container">
    <div class="header">Header</div>
    <div class="models" style="padding-top: 15px;">

        <ModelRadio bind:group={selectedModel} value="communes:v1" description="French town names"></ModelRadio>
        <ModelRadio bind:group={selectedModel} value="communes:v2" description="French town names"></ModelRadio>
        <ModelRadio bind:group={selectedModel} value="communes:v3" description="French town names"></ModelRadio>
        <ModelRadio bind:group={selectedModel} value="prénom:v1" description="French first names"></ModelRadio>
        <ModelRadio bind:group={selectedModel} value="prénom:v2" description="French first names"></ModelRadio>
        <ModelRadio bind:group={selectedModel} value="prénom:v3" description="French first names"></ModelRadio>


    </div>

    <div class="panel" style="padding-top: 15px;padding-left: 30px;padding-right: 30px;">
        <div class="mb-6">
            <Label for="default-input" class="mb-2 block">Start-text</Label>
            <Input id="default-input" placeholder="Default input"/>
        </div>
        <div class="max-h-85 overflow-y-auto border border-gray-200 p-4">
            <ul class="space-y-2">
                {#each items as item}
                    <li class="px-2 py-1 bg-gray-50 rounded">{item}</li>
                {/each}
            </ul>
        </div>
        <div class="flex mt-7">
            <GradientButton class="mr-6" color="blue">
                <Spinner class="me-3" size="4" color="gray" />
                Loading ...
            </GradientButton>
            <GradientButton class="mr-6" color="blue">
  Choose Plan <AnnotationSolid class="ms-2 h-5 w-5" />
            </GradientButton>
        </div>
    </div>
    <div class="parameters">Parameters</div>
    <div class="history">History</div>
    <div class="footer">Footer</div>
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

    /* --- Assignation des éléments à leurs zones --- */
    .header {
        grid-area: header;
    }

    .models {
        grid-area: models;
    }

    .panel {
        grid-area: panel;
    }

    .parameters {
        grid-area: parameters;
    }

    .history {
        grid-area: history;
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
        background-color: #0077b6;
    }

    .models {
        background-color: white;
        place-items: center;
    }

    .panel {
        background-color: white;
    }

    .parameters {
        background-color: #e9c46a;
    }

    .history {
        background-color: #f4a261;
    }

    .footer {
        background-color: #e76f51;
    }
</style>
