#!/usr/bin/env bash

# Source this file, then call configure_gaze_wam_nccl_transport before a
# multi-node launch. The verified H200-4102/4103 profile uses eight RoCE rails.
configure_gaze_wam_nccl_transport() {
  local transport="${NCCL_TRANSPORT:-roce}"
  local bootstrap_ifname="${NCCL_BOOTSTRAP_IFNAME:-net0}"
  local roce_hcas="${NCCL_ROCE_HCAS:-mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7}"
  local roce_gid_index="${NCCL_ROCE_GID_INDEX:-3}"
  local preflight="${NCCL_TRANSPORT_PREFLIGHT:-true}"

  case "$preflight" in
    true|false) ;;
    *) echo "NCCL_TRANSPORT_PREFLIGHT must be true or false, got: $preflight" >&2; return 2 ;;
  esac

  case "$transport" in
    roce)
      export NCCL_IB_DISABLE=0
      export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-$bootstrap_ifname}"
      export NCCL_IB_HCA="${NCCL_IB_HCA:-$roce_hcas}"
      export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-$roce_gid_index}"

      if [[ "$preflight" == true ]]; then
        if [[ ! -d "/sys/class/net/$NCCL_SOCKET_IFNAME" ]]; then
          echo "Missing NCCL bootstrap interface: $NCCL_SOCKET_IFNAME" >&2
          return 1
        fi

        local hca
        local gid_path
        local gid_type_path
        local gid
        local gid_type
        local hca_list="${NCCL_IB_HCA#=}"
        local -a hcas
        IFS=',' read -ra hcas <<< "$hca_list"
        for hca in "${hcas[@]}"; do
          gid_path="/sys/class/infiniband/$hca/ports/1/gids/$NCCL_IB_GID_INDEX"
          gid_type_path="/sys/class/infiniband/$hca/ports/1/gid_attrs/types/$NCCL_IB_GID_INDEX"
          if [[ ! -r "$gid_path" || ! -r "$gid_type_path" ]]; then
            echo "Missing GID $NCCL_IB_GID_INDEX for HCA $hca" >&2
            return 1
          fi
          gid="$(<"$gid_path")"
          gid_type="$(<"$gid_type_path")"
          if [[ "$gid" == "0000:0000:0000:0000:0000:0000:0000:0000" || "$gid_type" != "RoCE v2" ]]; then
            echo "HCA $hca GID $NCCL_IB_GID_INDEX is not a usable RoCE v2 address: $gid ($gid_type)" >&2
            return 1
          fi
        done
      fi

      echo "NCCL transport: RoCE v2; HCA=$NCCL_IB_HCA GID=$NCCL_IB_GID_INDEX bootstrap=$NCCL_SOCKET_IFNAME" >&2
      ;;
    socket)
      export NCCL_IB_DISABLE=1
      export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-$bootstrap_ifname}"
      echo "NCCL transport: socket; interface=$NCCL_SOCKET_IFNAME" >&2
      ;;
    inherit)
      echo "NCCL transport: inherited environment" >&2
      ;;
    *)
      echo "NCCL_TRANSPORT must be roce, socket, or inherit, got: $transport" >&2
      return 2
      ;;
  esac
}
