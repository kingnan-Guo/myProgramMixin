<template>
  <div class="main">
    <div class="container">
      <transition name="fade" mode="out-in">
        <keep-alive>
          <router-view></router-view>
        </keep-alive>
      </transition>
    </div>

  </div>
</template>

<script>
export default {
  name: 'mainIndex',
  components: {
  },
  data () {
    return {
      chatUnit: false,
      showMarquee: true,
      newCombatReadinessMarquee: []
    }
  },
  mounted () {
  }
}
</script>
