<script lang="ts">
  import { TextInput } from 'flowbite-svelte';

  export let models: string[] = [];
  export let selected: string;
  export { selected as bind:selected };

  let query = '';
  $: filtered = models.filter((m) => m.toLowerCase().includes(query.toLowerCase()));
</script>

<div class="flex flex-col h-full">
  <TextInput placeholder="search" bind:value={query} class="mb-2" />

  <div class="flex-1 overflow-y-auto space-y-1 pr-1">
    {#each filtered as model}
      <label class="flex items-center gap-2 cursor-pointer text-sm">
        <input type="radio" name="model" value={model} bind:group={selected} />
        <span>{model}</span>
      </label>
    {/each}

    {#if !filtered.length}
      <p class="text-gray-400 text-sm">Aucun modèle…</p>
    {/if}
  </div>
</div>