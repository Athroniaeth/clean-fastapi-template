<script>

    import {Helper, Radio} from "flowbite-svelte";


    export let value;
    export let description;
    export let checked = false;
    export let group = undefined;
</script>
<div style="margin-top: 10px">
    <Radio name="models" {value} bind:group checked={checked} color="secondary">
        {value}
    </Radio>
    <Helper class="ps-6" color="gray">
        {description}
    </Helper>
</div>