<script>
    import {Input, Label} from "flowbite-svelte";

    export let id;
    export let label;
    export let min = 0;
    export let max = 1;
    export let step = 0.01;
    export let value = 0;
    export let round = false;
</script>
<div class="mb-4">
    <div class="text-sm">
        <Label for={id} class="block">
            {label} :
            {#if round}
                {Math.round(value)}
            {:else}
                {typeof value === 'number' ? value.toFixed(2) : value}
            {/if}
        </Label>
    </div>
    <Input
            id={id}
            type="range"
            {min}
            {max}
            {step}
            bind:value
    />
</div>